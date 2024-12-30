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

interface ColumnMetadata {
  description: string;
  examples: string[];
  foreignKey?: {
    table: string;
    column: string;
  };
}

class OllamaMCPHost {
  private client: Client;
  private transport: StdioClientTransport;
  private modelName: string;
  private schemaCache: Map<string, DatabaseSchema[]> = new Map();
  private columnMetadata: Map<string, Map<string, ColumnMetadata>> = new Map();
  private chatHistory: { role: string; content: string }[] = [];
  private readonly MAX_HISTORY_LENGTH = 20;
  private readonly MAX_RETRIES = 5;

  private static readonly QUERY_GUIDELINES = `
When analyzing questions:
1. First write a SQL query to get the necessary information. Identify which tables contain the relevant information by looking at:
   - Table names and their purposes
   - Column names and descriptions
   - Foreign key relationships
2. Use the 'query' tool to execute the SQL query
3. If unsure about table contents, write a sample query first:
   SELECT column_name, COUNT(*) FROM table_name GROUP BY column_name LIMIT 5;
4. For complex questions, break down into multiple queries:
   - First query to validate data availability
   - Second query to get detailed information
5. Always include appropriate JOIN conditions when combining tables
6. Use WHERE clauses to filter irrelevant data
7. Consider using ORDER BY for sorted results

Important: Only use SELECT statements - no modifications allowed!

When you are finished, analyze the results and provide a natural language response.`;

  constructor(modelName?: string) {
    this.modelName =
      modelName || process.env.OLLAMA_MODEL || "qwen2.5-coder:7b-instruct";
    this.transport = new StdioClientTransport({
      command: "npx",
      args: ["-y", "@modelcontextprotocol/server-postgres", databaseUrl!],
    });
    this.client = new Client(
      { name: "ollama-mcp-host", version: "1.0.0" },
      { capabilities: {} }
    );
  }

  private async detectTableRelationships(): Promise<void> {
    // Query the database to find foreign key relationships
    const sql = `
      SELECT
        tc.table_name as table_name,
        kcu.column_name as column_name,
        ccu.table_name AS foreign_table_name,
        ccu.column_name AS foreign_column_name
      FROM information_schema.table_constraints tc
      JOIN information_schema.key_column_usage kcu
        ON tc.constraint_name = kcu.constraint_name
      JOIN information_schema.constraint_column_usage ccu
        ON ccu.constraint_name = tc.constraint_name
      WHERE constraint_type = 'FOREIGN KEY'
    `;

    try {
      const result = await this.executeQuery(sql);
      const relationships = JSON.parse(result);

      // Create initial metadata for foreign keys
      relationships.forEach((rel: any) => {
        const tableMetadata =
          this.columnMetadata.get(rel.table_name) || new Map();

        tableMetadata.set(rel.column_name, {
          description: `Foreign key referencing ${rel.foreign_table_name}.${rel.foreign_column_name}`,
          examples: [],
          foreignKey: {
            table: rel.foreign_table_name,
            column: rel.foreign_column_name,
          },
        });

        this.columnMetadata.set(rel.table_name, tableMetadata);
      });
    } catch (error) {
      console.error("Error detecting table relationships:", error);
    }
  }

  private buildSystemPrompt(includeErrorContext: string = ""): string {
    let prompt =
      "You are a data analyst assistant. You have access to a PostgreSQL database with these tables:\n\n";

    // Add detailed schema information
    for (const [tableName, schema] of this.schemaCache.entries()) {
      prompt += `Table: ${tableName}\n`;
      prompt += "Columns:\n";

      for (const column of schema) {
        const metadata = this.columnMetadata
          .get(tableName)
          ?.get(column.column_name);
        prompt += `- ${column.column_name} (${column.data_type})`;

        if (metadata) {
          prompt += `: ${metadata.description}`;
          if (metadata.foreignKey) {
            prompt += ` [References ${metadata.foreignKey.table}.${metadata.foreignKey.column}]`;
          }
        }
        prompt += "\n";
      }
      prompt += "\n";
    }

    // Add query guidelines
    prompt += "\nQuery Guidelines:\n";
    prompt += OllamaMCPHost.QUERY_GUIDELINES;

    if (includeErrorContext) {
      prompt += `\nPrevious Error Context: ${includeErrorContext}\n`;
      prompt +=
        "Please revise your approach and try a different query strategy.\n";
    }

    return prompt;
  }

  async connect() {
    await this.client.connect(this.transport);

    // First detect relationships
    await this.detectTableRelationships();

    // Then load schemas
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
        }
      }
    }
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

  private addToHistory(role: string, content: string) {
    this.chatHistory.push({ role, content });
    while (this.chatHistory.length > this.MAX_HISTORY_LENGTH) {
      this.chatHistory.shift();
    }
  }

  async processQuestion(question: string): Promise<string> {
    try {
      let attemptCount = 0;
      let lastError: string | undefined;

      while (attemptCount <= this.MAX_RETRIES) {
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

        // Get response from Ollama
        const response = await ollama.chat({
          model: this.modelName,
          messages: messages,
        });

        // Extract SQL query
        const sqlMatch = response.message.content.match(
          /```sql\n([\s\S]*?)\n```/
        );
        if (!sqlMatch) {
          return response.message.content;
        }

        const sql = sqlMatch[1].trim();
        console.log("Executing SQL:", sql);

        try {
          // Execute the query
          const queryResult = await this.executeQuery(sql);
          this.addToHistory("assistant", response.message.content);

          // Ask for result interpretation
          const interpretationMessages = [
            ...messages,
            { role: "assistant", content: response.message.content },
            {
              role: "user",
              content: `Here are the results of the SQL query: ${queryResult}\n\nPlease analyze these results and provide a clear summary.`,
            },
          ];

          const finalResponse = await ollama.chat({
            model: this.modelName,
            messages: interpretationMessages,
          });

          this.addToHistory("assistant", finalResponse.message.content);
          return finalResponse.message.content;
        } catch (error) {
          lastError = error instanceof Error ? error.message : String(error);
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
