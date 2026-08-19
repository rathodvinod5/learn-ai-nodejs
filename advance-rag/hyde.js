import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import "dotenv/config";

const rawDocs = [
  new Document({
    pageContent: `To resolve an HTTP 404 error on Nginx,
          verify your root directive in nginx.conf matches your actual build directory path`,
    metadata: { id: 1 },
  }),
  new Document({
    pageContent: `Node.js memory leaks can often be tracked using the heapdump module or 
    inspecting the V8 heap profile via Chrome DevTools.`,
    metadata: { id: 2 },
  }),
];

async function main() {
  const embedding = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
  });
  const llm = new ChatOpenAI({
    model: "gpt-4o-mini",
    temperature: 0,
    apiKey: process.env.OPENAI_API_KEY,
  });
  const vectorStore = await MemoryVectorStore.fromDocuments(rawDocs, embedding);
  const baseRetriever = vectorStore.asRetriever({ k: 1 });
  const hydePrompt = ChatPromptTemplate.fromTemplate(
    `Please write a short, highly technical paragraph ansering this question.
    Do not say 'Here is the answer:', just write the answer directly
    
    Question:
    {question}`,
  );
  const query =
    "My nginx server is showing a missing page error, how do I configure the path?";
  const hydePromptChain = await RunnableSequence.from([
    hydePrompt,
    llm,
    new StringOutputParser(),
  ]);
  const hydeDoc = await hydePromptChain.invoke({
    question: query,
  });

  //   Alternative way, instead of RunnableSequence
  //   const messages = [
  //     new SystemMessage(
  //       `You are an expert technical assistant. Your job is to write a hypothetical
  //         highly detailed paragraph answering the user's question. Do not include any introductory
  //         phrases like 'Here is the answer'. Write the technical document directly.`,
  //     ),
  //     new HumanMessage(query),
  //   ];
  //   const response = await llm.invoke(messages);
  //   const hydeDoc = response.content;

  console.log("HydeDoc: ", hydeDoc);
  //   const result = await vectorStore.similaritySearch(hydeDoc);
  const result = await baseRetriever.invoke(hydeDoc);
  console.log("Result: ", result);
}

main().catch((error) => console.log("Err: ", error));
