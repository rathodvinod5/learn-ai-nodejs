import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import "dotenv/config";

const embedding = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
});
const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
  apiKey: process.env.OPENAI_API_KEY,
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
  const query = "What chip does the original Apple Vision Pro have?";
  const answer1 = await runCorrectiveRAG(query, vectorStore);
  console.log("\nFinal Output:\n", answer1);

  const query2 =
    "When is the Apple Vision Pro 2 releasing and what chip will it use?";
  const answer2 = await runCorrectiveRAG(query2, vectorStore);
  console.log(`\nFinal Output:\n`, answer2);
}

async function runCorrectiveRAG(userQuery, vectorStore) {
  const baseRetriever = vectorStore.asRetriever(2);
  const resultDocs = await baseRetriever.invoke(userQuery);

  let context = resultDocs.map((doc) => doc.pageContent).join("\n");
  const status = await validateResult(context, userQuery);

  //   console.log("context: ", context);
  if (resultDocs.length <= 0 && !status) {
    const searchResult = await mockWebSearch(userQuery);
    context = `[Local Documents (Insufficient)]\n
      ${retrievedDocs.map((doc) => doc.pageContent).join("\n")}\n\n
      [Web Search Discovery]\n${searchResult}`;
  }

  const generationPrompt = ChatPromptTemplate.fromTemplate(
    `Answer the user query based strictly on the provided context. 
    If the context contains both local data and web discoveries, synthesize them seamlessly.
    
    Context:
    {context}
    
    Query: 
    {query}

    Answer:`,
  );
  const generationChain = RunnableSequence.from([
    generationPrompt,
    llm,
    new StringOutputParser(),
  ]);
  const result = await generationChain.invoke({
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
    `You are an accurate quality control grader, Evaluat if the provided context contains
    sufficient information to answer the query.
    
    Context:
    {context}
    
    User query:
    {query}
       
    Respond with exactly one word: "YES" if the context is highly relevant and sufficient, 
    or "NO" if it is insufficient or missing key information.
    `,
  );
  const validationChain = RunnableSequence.from([
    validationPrompt,
    llm,
    new StringOutputParser(),
  ]);
  const result = await validationChain.invoke({
    context: context,
    query: query,
  });
  console.log("result: ", result);
  return result;
}

main();
