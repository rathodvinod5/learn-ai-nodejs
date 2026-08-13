import "dotenv/config";

import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

// --- 1. Sample documents ---
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
  (item, i) => new Document({ pageContent: item, metadata: { index: i } }),
);

// --- 2. Prompt used to generate query variations ---
const queryVariation = ChatPromptTemplate.fromTemplate(
  `Your are an expert AI assistant. Yout task to create 3 different variations of 
  given user question to retrieve relevant documents from vector database. Provide
  these alternative questions seperated by new lines with no numbering or extra commenting.

  Original question:
  {question}
  `,
);

async function generateQueries(llm, question) {
  const chain = queryVariation.pipe(llm).pipe(new StringOutputParser());
  const raw = await chain.invoke({ question });
  const variations = raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  return [question, ...variations];
}

// --- 3. Run retrieval for each query and dedupe results ---
async function multiQueryRetrieve(vectorStore, queries, k = 2) {
  const resultsPerQuery = await Promise.all(
    queries.map((q) => vectorStore.similaritySearch(q, k)),
  );

  const seen = new Set();
  const uniqueDocs = [];

  for (const docsForQuery of resultsPerQuery) {
    for (const doc of docsForQuery) {
      const key = doc.pageContent; // dedupe by content (or use metadata.id)
      if (!seen.has(key)) {
        seen.add(key);
        uniqueDocs.push(doc);
      }
    }
  }

  return uniqueDocs;
}

async function main() {
  const llm = new ChatOpenAI({
    model: "gpt-4o-mini",
    temperature: 0,
  });
  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
  });
  const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);

  // const userQuery = "How do I stop paying for my account?";
  const userQuery = "Does the phone works in offline mode";
  const queries = await generateQueries(llm, userQuery);
  console.log(`\nOriginal query: "${userQuery}"\n`);

  console.log("Generated queries:");
  queries.forEach((q, i) => console.log(`  ${i + 1}. ${q}`));

  const results = await multiQueryRetrieve(vectorStore, queries, 2);

  console.log(`\nRetrieved ${results.length} unique chunks:\n`);
  results.forEach((doc, i) => console.log(`${i + 1}. ${doc.pageContent}`));
}
main();
