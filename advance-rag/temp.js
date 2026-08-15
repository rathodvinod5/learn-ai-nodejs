import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { InMemoryStore } from "@langchain/core/stores";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import "dotenv/config";

const rawDocs = [
  "Dashoboard contains quick links to all the other pages/screens",
  "The profile page shows users spend details",
  "To cancel your subscription, go to Account Settings > Billing > Manage Plan.",
  "You can terminate your account plan at any time from the billing dashboard.",
  "Refunds are processed within 5-7 business days after cancellation.",
  "Our mobile app supports offline mode for downloaded content.",
  "Video transcripts are auto-generated using Whisper and stored with timestamps.",
  "Enterprise plans include SSO, audit logs, and dedicated support.",
];
const docs = rawDocs.map(
  (item) =>
    new Document({
      pageContent: item,
      metadata: { index: 1 },
    }),
);

async function multiQueryRetrieve(vectorStore, queries) {
  queries = queries.split("\n").map((item) => item.trim());
  //   console.log(JSON.stringify(queries));
  const result = await Promise.all(
    queries.map((item) => vectorStore.similaritySearch(item, 2)),
  );

  const seen = new Set();
  const uniqueDocs = [];
  for (const docs of result) {
    for (const item of docs) {
      const { pageContent } = item;
      if (!seen.has(pageContent)) {
        seen.add(pageContent);
        uniqueDocs.push(pageContent);
      }
    }
  }

  return uniqueDocs;
}

async function main() {
  const embedding = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
  });
  const llm = new ChatOpenAI({
    model: "gpt-4o-mini",
    apiKey: process.env.OPENAI_API_KEY,
    temperature: 0,
  });
  const vectorStore = await MemoryVectorStore.fromDocuments(docs, embedding);
  //   const baseRetiever = await vectorStore.asRetriever({ k: 3 });

  const queryVariation = ChatPromptTemplate.fromTemplate(
    `You are an expert AI assistent. Your task is to create 3 different version of the
    given user question to retirieve relevant documents from vector database. Provide
    these alternative questions seperated by newline with no numbering or extra commenting.

    Question:
    {question}
    `,
  );
  const userQuery = "Does the phone works in offline mode";
  //   const queryVariations = await generateQueries(userQuery);
  const queryChain = await RunnableSequence.from([
    queryVariation,
    llm,
    new StringOutputParser(),
  ]);
  const queries = await queryChain.invoke({
    question: userQuery,
  });
  console.log("queries: \n", queries);
  const result = await multiQueryRetrieve(vectorStore, queries);
  console.log("Unique queries are: \n", result);
}
main();
