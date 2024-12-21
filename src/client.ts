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
  const resources = await client.request(
    { method: "resources/list" },
    ListResourcesResultSchema
  );
  console.log("Resources:", resources);

  // Read a specific resource
  // const resourceContent = await client.request(
  //   {
  //     method: "resources/read",
  //     params: {
  //       uri: "file:///example.txt",
  //     },
  //   },
  //   ReadResourceResultSchema
  // );
}
