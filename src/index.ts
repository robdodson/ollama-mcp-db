import ollama from "ollama";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import {
  ReadResourceResultSchema,
  ListResourcesResultSchema,
  CallToolResultSchema,
} from "@modelcontextprotocol/sdk/types.js";
import dotenv from "dotenv";
import { fileURLToPath } from "url";

// Load environment variables from .env file
dotenv.config();

// Check for required environment variables
const databaseUrl = process.env.DATABASE_URL;
if (!databaseUrl) {
  console.error("Error: DATABASE_URL not found in environment or .env file");
  console.error("Please create a .env file with the following format:");
  console.error("DATABASE_URL=postgres://user:password@localhost:5432/dbname");
  process.exit(1);
}

interface DatabaseSchema {
  column_name: string;
  data_type: string;
}

class OllamaMCPHost {
  private client: Client;
  private transport: StdioClientTransport;
  private modelName: string;
  private schemaCache: Map<string, DatabaseSchema[]> = new Map();
  private chatHistory: { role: string; content: string }[] = [];
  private readonly MAX_HISTORY_LENGTH = 20;
  private readonly MAX_RETRIES = 2; // Maximum number of retry attempts

  constructor(modelName?: string) {
    this.modelName =
      modelName || process.env.OLLAMA_MODEL || "qwen2.5-coder:7b-instruct";

    this.transport = new StdioClientTransport({
      command: "npx",
      args: ["-y", "@modelcontextprotocol/server-postgres", databaseUrl!],
    });

    this.client = new Client(
      {
        name: "ollama-mcp-host",
        version: "1.0.0",
      },
      {
        capabilities: {},
      }
    );
  }

  private addToHistory(role: string, content: string) {
    this.chatHistory.push({ role, content });

    while (this.chatHistory.length > this.MAX_HISTORY_LENGTH) {
      this.chatHistory.splice(0, 2);
    }
  }

  async connect() {
    await this.client.connect(this.transport);

    const resources = await this.client.request(
      { method: "resources/list" },
      ListResourcesResultSchema
    );

    for (const resource of resources.resources) {
      if (resource.uri.endsWith("/schema")) {
        const schema = await this.client.request(
          {
            method: "resources/read",
            params: { uri: resource.uri },
          },
          ReadResourceResultSchema
        );

        if (schema.contents[0]?.text) {
          try {
            const tableName = resource.uri.split("/").slice(-2)[0];
            this.schemaCache.set(
              tableName,
              JSON.parse(schema.contents[0].text as string)
            );
          } catch (error) {
            console.error(
              `Failed to parse schema for resource ${resource.uri}:`,
              error instanceof Error ? error.message : String(error)
            );
          }
        } else {
          console.warn(`No text content found for resource ${resource.uri}`);
        }
      }
    }
  }

  private buildSystemPrompt(includeErrorContext: string = ""): string {
    let prompt =
      "You are a data analyst assistant. You have access to a PostgreSQL database with the following tables and schemas:\n\n";

    for (const [tableName, schema] of this.schemaCache.entries()) {
      prompt += `Table: ${tableName}\nColumns:\n`;
      for (const column of schema) {
        prompt += `- ${column.column_name} (${column.data_type})\n`;
      }
      prompt += "\n";
    }

    prompt += "\nWhen answering questions about the data:\n";
    prompt += "1. First write a SQL query to get the necessary information\n";
    prompt += "2. Use the 'query' tool to execute the SQL query\n";
    prompt +=
      "3. Analyze the results and provide a natural language response\n";
    prompt +=
      "\nImportant: Only use SELECT statements - no modifications allowed.\n";

    if (includeErrorContext) {
      prompt += `\nThe previous query attempt failed with the following error: ${includeErrorContext}\n`;
      prompt +=
        "Please revise your approach and try a different query strategy.\n";
    }

    return prompt;
  }

  private async executeQuery(sql: string): Promise<string> {
    const response = await this.client.request(
      {
        method: "tools/call",
        params: {
          name: "query",
          arguments: { sql },
        },
      },
      CallToolResultSchema
    );

    if (!response.content?.[0]?.text) {
      throw new Error("No text content received from query");
    }
    return response.content[0].text as string;
  }

  private async attemptQuery(
    messages: { role: string; content: string }[]
  ): Promise<{
    success: boolean;
    response: string;
    sql?: string;
    queryResult?: string;
    error?: string;
  }> {
    const response = await ollama.chat({
      model: this.modelName,
      messages: messages,
    });

    const sqlMatch = response.message.content.match(/```sql\n([\s\S]*?)\n```/);
    if (!sqlMatch) {
      return {
        success: false,
        response: response.message.content,
        error: "No SQL query found in response",
      };
    }

    const sql = sqlMatch[1].trim();
    console.log("Executing SQL:", sql);

    try {
      const queryResult = await this.executeQuery(sql);
      return {
        success: true,
        response: response.message.content,
        sql,
        queryResult,
      };
    } catch (error) {
      return {
        success: false,
        response: response.message.content,
        sql,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  async processQuestion(question: string): Promise<string> {
    try {
      let attemptCount = 0;
      let lastError: string | undefined;

      while (attemptCount <= this.MAX_RETRIES) {
        // Prepare messages for this attempt
        const messages = [
          { role: "system", content: this.buildSystemPrompt(lastError) },
          ...this.chatHistory,
          { role: "user", content: question },
        ];

        if (attemptCount === 0) {
          this.addToHistory("user", question);
        }

        console.log(
          attemptCount > 0 ? `\nRetry attempt ${attemptCount}...` : ""
        );

        const result = await this.attemptQuery(messages);

        if (result.success && result.queryResult) {
          // Query succeeded
          this.addToHistory("assistant", result.response);

          // Request interpretation of results
          const resultPrompt = `Here are the results of the SQL query: ${result.queryResult}\n\nPlease analyze these results and provide a clear summary.`;
          this.addToHistory("user", resultPrompt);

          const finalResponse = await ollama.chat({
            model: this.modelName,
            messages: [
              ...messages,
              { role: "assistant", content: result.response },
              { role: "user", content: resultPrompt },
            ],
          });

          this.addToHistory("assistant", finalResponse.message.content);
          return finalResponse.message.content;
        } else {
          // Query failed
          lastError = result.error;
          if (attemptCount === this.MAX_RETRIES) {
            return `I apologize, but I was unable to successfully query the database after ${
              this.MAX_RETRIES + 1
            } attempts. The last error was: ${lastError}`;
          }
        }

        attemptCount++;
      }

      return "An unexpected error occurred while processing your question.";
    } catch (error) {
      console.error("Error processing question:", error);
      return `An error occurred: ${
        error instanceof Error ? error.message : String(error)
      }`;
    }
  }

  async cleanup() {
    await this.transport.close();
  }
}

async function main() {
  const host = new OllamaMCPHost();
  const readline = (await import("readline")).default.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  try {
    await host.connect();
    console.log(
      "\nConnected to database. You can now ask questions about your data."
    );
    console.log('Type "exit" to quit.\n');

    const askQuestion = (prompt: string) =>
      new Promise<string>((resolve) => {
        readline.question(prompt, resolve);
      });

    while (true) {
      const userInput = await askQuestion(
        "\nWhat would you like to know about your data? "
      );

      if (userInput.toLowerCase() === "exit") {
        console.log("\nGoodbye!\n");
        readline.close();
        await host.cleanup();
        process.exit(0);
      }

      console.log("\nAnalyzing...\n");
      const answer = await host.processQuestion(userInput);
      console.log("\n", answer, "\n");
    }
  } catch (error) {
    console.error(
      "Error:",
      error instanceof Error ? error.message : String(error)
    );
    readline.close();
    await host.cleanup();
    process.exit(1);
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch(console.error);
}

export default OllamaMCPHost;
