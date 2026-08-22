import "dotenv/config";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";

const embedding = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
});
const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
  apiKey: process.env.OPENAI_API_KEY,
  temperature: 0,
});

async function main() {
  const rawDocs = [
    new Document({
      pageContent:
        "The original Apple Vision Pro launched in early 2024 featuring micro-OLED displays and an M2 chip.",
    }),
    new Document({
      pageContent:
        "LangChain is a popular open-source framework designed to simplify building applications with LLMs.",
    }),
    new Document({
      pageContent:
        "Node.js allows developers to write server-side JavaScript applications outside the browser environment.",
    }),
  ];
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });
  const splittedDocs = await textSplitter.splitDocuments(rawDocs);
  const vectorStore = await MemoryVectorStore.fromDocuments(
    splittedDocs,
    embedding,
  );
  const baseRetriever = vectorStore.asRetriever(2);

  const query = "What chip does the original Apple Vision Pro have?";
  let answer1 = await runCorrectiveRAG(query, baseRetriever);
  console.log("\nfinal Result: \n", answer1, "\n");

  const query2 =
    "When is the Apple Vision Pro 2 releasing and what chip will it use?";
  const answer2 = await runCorrectiveRAG(query2, baseRetriever);
  console.log(`\nFinal Output:\n${answer2}\n`);
}

async function runCorrectiveRAG(userQuery, baseRetriever) {
  const documents = await baseRetriever.invoke(userQuery);
  let context = documents.map((docs) => docs.pageContent).join("\n");

  const status = await validateResult(context, userQuery);
  if (documents.length <= 0 && !status) {
    context = await mockWebSearch(userQuery);
  }

  const finalPrompt = ChatPromptTemplate.fromTemplate(
    `You are an AI assitant, please answer the user query only by using context given below.
    
    Context:
    {context}

    User query:
    {query}
    `,
  );
  const finalChain = RunnableSequence.from([
    finalPrompt,
    llm,
    new StringOutputParser(),
  ]);
  const result = await finalChain.invoke({
    context: context,
    query: userQuery,
  });

  return result;
}

async function mockWebSearch(query) {
  console.log(`🔍 [CRAG] Triggering external web search for: "${query}"...`);
  if (query.toLowerCase().includes("apple vision pro 2")) {
    return "The Apple Vision Pro 2 is rumored to release in late 2026 with an upgraded M5 chip and lighter design.";
  }
  return "No external search results found.";
}

async function validateResult(context, query) {
  const validationPrompt = ChatPromptTemplate.fromTemplate(
    `You are an accurate quality control grader, Evaluate if the provided context 
    contains sufficient information to answer the user query
    
    Context:
    {context}
    
    User query:
    {query}
    
    Respond with exactly one word: "YES" if the context is highly relevant 
    and sufficient, or "NO" if it is insufficient or missing key information.
    `,
  );
  const validationChain = RunnableSequence.from([
    validationPrompt,
    llm,
    new StringOutputParser(),
  ]);
  const status = await validationChain.invoke({
    context: context,
    query: query,
  });
  return status?.toLocaleLowerCase() === "yes";
}

main();

//          +----------------------------------+
//          |         User Query               |
//          +----------------+-----------------+
//                           |
//                           v
//          +----------------+-----------------+
//          |   Retrieve Local Documents       |
//          +----------------+-----------------+
//                           |
//                           v
//          +----------------+-----------------+
//          | Evaluator (LLM Grading Relevance)|
//          +----------------+-----------------+
//                           |
//          +----------------+-----------------+
//          | Are documents sufficient?        |
//          +-------+------------------+-------+

//                  |                  |
//         YES (Accurate)         NO (Inaccurate/Missing)

//                  |                  |
//                  |                  v
//                  |       +----------+-----------+
//                  |       |  Web Search Trigger  |
//                  |       | (Fallback/Knowledge) |
//                  |       +----------+-----------+
//                  |                  |
//                  v                  v
//          +-------+------------------+-------+
//          |  Synthesize Consolidated Context |
//          +----------------+-----------------+
//                           |
//                           v
//          +----------------+-----------------+
//          |     Generate Final Answer        |
//          +----------------------------------+
