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
  console.error(
    "DATABASE_URL=postgresql://user:password@localhost:5432/dbname"
  );
  process.exit(1);
}

// Type definition for database schema information
interface DatabaseSchema {
  column_name: string;
  data_type: string;
}

/**
 * OllamaMCPHost combines Ollama's LLM capabilities with MCP's database access.
 * It allows natural language queries against a PostgreSQL database by:
 * 1. Understanding the database schema
 * 2. Generating appropriate SQL queries using Ollama
 * 3. Executing these queries through the MCP PostgreSQL server
 * 4. Interpreting the results using Ollama
 */
class OllamaMCPHost {
  private client: Client; // MCP client for connecting to the Postgres server
  private transport: StdioClientTransport; // Transport layer for MCP communication
  private modelName: string; // Name of the Ollama model to use
  private schemaCache: Map<string, DatabaseSchema[]> = new Map(); // Cache of database structure

  /**
   * Creates a new OllamaMCPHost instance
   * @param modelName - The Ollama model to use (defaults to 'qwen2.5-coder:7b-instruct')
   */
  constructor(modelName?: string) {
    // Use provided modelName, or OLLAMA_MODEL from .env, or fall back to 'qwen2.5-coder:7b-instruct'
    this.modelName =
      modelName || process.env.OLLAMA_MODEL || "qwen2.5-coder:7b-instruct";

    // Initialize the MCP transport layer to communicate with the Postgres server
    // The server is started using npx for convenience
    this.transport = new StdioClientTransport({
      command: "npx",
      args: ["-y", "@modelcontextprotocol/server-postgres", databaseUrl!],
    });

    // Create the MCP client with basic configuration
    this.client = new Client(
      {
        name: "ollama-mcp-host",
        version: "1.0.0",
      },
      {
        capabilities: {}, // No special capabilities needed for basic querying
      }
    );
  }

  /**
   * Establishes connections and loads database schema information.
   * Must be called before processing any questions.
   */
  async connect() {
    // Connect to the MCP server
    await this.client.connect(this.transport);

    // Get list of available resources (tables) from the database
    const resources = await this.client.request(
      { method: "resources/list" },
      ListResourcesResultSchema
    );

    // Load and cache schema information for each table
    // This will be used to provide context to the LLM
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

  /**
   * Builds a detailed system prompt for the LLM that includes:
   * - Complete database schema information
   * - Instructions for generating SQL queries
   * - Guidelines for data analysis
   */
  private buildSystemPrompt(): string {
    let prompt =
      "You are a data analyst assistant. You have access to a PostgreSQL database with the following tables and schemas:\n\n";

    // Include detailed schema information for each table
    for (const [tableName, schema] of this.schemaCache.entries()) {
      prompt += `Table: ${tableName}\nColumns:\n`;
      for (const column of schema) {
        prompt += `- ${column.column_name} (${column.data_type})\n`;
      }
      prompt += "\n";
    }

    // Add instructions for the LLM
    prompt += "\nWhen answering questions about the data:\n";
    prompt += "1. First write a SQL query to get the necessary information\n";
    prompt += "2. Use the 'query' tool to execute the SQL query\n";
    prompt +=
      "3. Analyze the results and provide a natural language response\n";
    prompt +=
      "\nImportant: Only use SELECT statements - no modifications allowed.\n";

    return prompt;
  }

  /**
   * Executes a SQL query through the MCP server's query tool
   * @param sql - The SQL query to execute
   * @returns The query results as a string
   */
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

  /**
   * Processes a natural language question about the database
   * @param question - The question to answer
   * @returns A natural language response based on the data
   */
  async processQuestion(question: string): Promise<string> {
    try {
      // Set up the conversation with Ollama
      const messages = [
        { role: "system", content: this.buildSystemPrompt() },
        { role: "user", content: question },
      ];

      // Get initial response from Ollama (should include SQL query)
      let response = await ollama.chat({
        model: this.modelName,
        messages: messages,
      });

      // Look for SQL queries in the response (enclosed in SQL code blocks)
      const sqlMatch = response.message.content.match(
        /```sql\n([\s\S]*?)\n```/
      );
      if (sqlMatch) {
        const sql = sqlMatch[1].trim();
        console.log("Executing SQL:", sql);

        // Execute the query through MCP
        const queryResult = await this.executeQuery(sql);

        // Ask Ollama to interpret the results
        messages.push({ role: "assistant", content: response.message.content });
        messages.push({
          role: "user",
          content: `Here are the results of the SQL query: ${queryResult}\n\nPlease analyze these results and provide a clear summary.`,
        });

        response = await ollama.chat({
          model: this.modelName,
          messages: messages,
        });
      }

      return response.message.content;
    } catch (error) {
      console.error("Error processing question:", error);
      return `An error occurred: ${
        error instanceof Error ? error.message : String(error)
      }`;
    }
  }

  /**
   * Cleans up resources and closes connections
   */
  async cleanup() {
    await this.transport.close();
  }
}

// Interactive chat interface
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

    // Promisify readline.question
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
  } finally {
    readline.close();
    await host.cleanup();
  }
}

// Only run if called directly
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch(console.error);
}

export default OllamaMCPHost;
