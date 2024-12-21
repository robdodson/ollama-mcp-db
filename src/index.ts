import ollama from "ollama";
import readline from "readline";
import { connect } from "./client.js";

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

async function askQuestion(query: string): Promise<string> {
  return new Promise((resolve) => {
    rl.question(query, (answer) => {
      resolve(answer);
    });
  });
}

async function chat() {
  const messages: { role: "user" | "assistant"; content: string }[] = [];

  try {
    while (true) {
      // Get user input
      const userInput = await askQuestion(
        "\nEnter your message (or 'exit' to quit): "
      );

      if (userInput.toLowerCase() === "exit") {
        console.log("\nGoodbye!");
        rl.close();
        break;
      }

      // Add user message to history
      messages.push({ role: "user", content: userInput });

      // Get AI response
      const response = await ollama.chat({
        model: "llama3.2",
        messages: messages,
        stream: true,
      });

      process.stdout.write("\nAI: ");
      let fullResponse = "";

      // Stream the response
      for await (const part of response) {
        process.stdout.write(part.message.content);
        fullResponse += part.message.content;
      }

      // Add a newline after the response
      process.stdout.write("\n");

      // Add AI response to message history
      messages.push({ role: "assistant", content: fullResponse });
    }
  } catch (error) {
    console.error("An error occurred:", error);
    rl.close();
  }
}

// Start the chat
console.log(
  "Welcome to the Interactive Chat! Type 'exit' to end the conversation."
);
// chat();
connect();
