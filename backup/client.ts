import {
  CallToolResultSchema,
  ListToolsResultSchema,
} from "@modelcontextprotocol/sdk/types.js";

export async function connect() {
  const { Client } = await import("@modelcontextprotocol/sdk/client/index.js");
  const { StdioClientTransport } = await import(
    "@modelcontextprotocol/sdk/client/stdio.js"
  );
  const { ListResourcesResultSchema, ReadResourceResultSchema } = await import(
    "@modelcontextprotocol/sdk/types.js"
  );

  const transport = new StdioClientTransport({
    command: "npx",
    args: [
      "-y",
      "@modelcontextprotocol/server-postgres",
      "postgresql://postgres:chinook@localhost/chinook",
    ],
  });

  const client = new Client(
    {
      name: "llama-chat-client",
      version: "1.0.0",
    },
    {
      capabilities: {},
    }
  );

  await client.connect(transport);

  // List available resources
  // https://modelcontextprotocol.io/docs/concepts/resources#resource-discovery
  // const resources = await client.request(
  //   { method: "resources/list" },
  //   ListResourcesResultSchema
  // );
  // console.log("Resources:", resources);

  // Read a specific resource
  // https://modelcontextprotocol.io/docs/concepts/resources#reading-resources
  // const resource = await client.request(
  //   { method: "resources/read", params: { uri: resources.resources[0].uri } },
  //   ReadResourceResultSchema
  // );
  // console.log("Resource:", resource);

  // List available tools
  // https://modelcontextprotocol.io/docs/concepts/tools#overview
  // const tools = await client.request(
  //   { method: "tools/list" },
  //   ListToolsResultSchema
  // );
  // console.log("Tools:", JSON.stringify(tools, null, 2));

  // Call tool
  // https://modelcontextprotocol.io/docs/concepts/tools#overview
  const result = await client.request(
    {
      method: "tools/call",
      params: {
        name: "query",
        arguments: { sql: `SELECT name FROM artist LIMIT 5;` },
      },
    },
    CallToolResultSchema
  );
  console.log("Result:", JSON.stringify(result, null, 2));
}
