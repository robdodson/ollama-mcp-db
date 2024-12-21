# Ollama MCP Database Assistant

An interactive chat interface that combines Ollama's LLM capabilities with PostgreSQL database access through the Model Context Protocol (MCP). Ask questions about your data in natural language and get AI-powered responses backed by real SQL queries.

## Features

- Natural language interface to your PostgreSQL database
- Automatic SQL query generation
- Schema-aware responses
- Interactive chat interface
- Secure, read-only database access

## Prerequisites

- Node.js 16 or higher
- A running PostgreSQL database
- [Ollama](https://ollama.ai) installed and running locally
- The qwen2.5-coder:7b-instruct model pulled in Ollama

## Setup

1. Clone the repository:

```bash
git clone [your-repo-url]
cd [your-repo-name]
```

2. Install dependencies:

```bash
npm install
```

3. Pull the required Ollama model:

```bash
ollama pull qwen2.5-coder:7b-instruct
```

4. Create a `.env` file in the project root:

```bash
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
OLLAMA_MODEL=qwen2.5-coder:7b-instruct  # Optional - this is the default
```

## Usage

1. Start the chat interface:

```bash
npm start
```

2. Ask questions about your data in natural language:

```
Connected to database. You can now ask questions about your data.
Type "exit" to quit.

What would you like to know about your data? Which products generated the most revenue last month?
Analyzing...

[AI will generate and execute a SQL query, then explain the results]
```

3. Type 'exit' to quit the application.

## How It Works

1. The application connects to your PostgreSQL database through MCP
2. It loads and caches your database schema
3. When you ask a question:
   - The schema and question are sent to Ollama
   - Ollama generates an appropriate SQL query
   - The query is executed through MCP
   - Results are sent back to Ollama for interpretation
   - You receive a natural language response

## Environment Variables

| Variable     | Description                  | Default                   |
| ------------ | ---------------------------- | ------------------------- |
| DATABASE_URL | PostgreSQL connection string | Required                  |
| OLLAMA_MODEL | Ollama model to use          | qwen2.5-coder:7b-instruct |

## Security

- All database access is read-only
- SQL queries are restricted to SELECT statements
- Database credentials are kept secure in your .env file

## Development

Built with:

- TypeScript
- Model Context Protocol (MCP)
- Ollama
- PostgreSQL

## Troubleshooting

### Common Issues

1. "Failed to connect to database"

   - Check your DATABASE_URL in .env
   - Verify PostgreSQL is running
   - Check network connectivity

2. "Failed to connect to Ollama"

   - Ensure Ollama is running (`ollama serve`)
   - Verify the model is installed (`ollama list`)

3. "Error executing query"
   - Check database permissions
   - Verify table/column names in the schema

## License

MIT

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request
