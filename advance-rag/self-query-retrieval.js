// npm install langchain @langchain/openai @langchain/classic
import "dotenv/config";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";

import { SelfQueryRetriever } from "@langchain/classic/retrievers/self_query";
import { FunctionalTranslator } from "@langchain/classic/retrievers/self_query/functional";
import { Document } from "@langchain/core/documents";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

const attributeInfo = [
  {
    name: "sourceType",
    description: "The type of source the chunk came from",
    type: "string or array of strings, one of pdf, video, audio, srt",
  },
  {
    name: "documentTitle",
    description: "Title of the parent document/lecture/video",
    type: "string",
  },
  {
    name: "timestampSeconds",
    description:
      "Timestamp in seconds where this chunk starts, for video/audio sources",
    type: "number",
  },
  {
    name: "pageNumber",
    description: "Page number, for PDF sources",
    type: "number",
  },
];

const docs = [
  new Document({
    pageContent:
      "The Q3 pricing strategy focuses on tiered plans for enterprise customers.",
    metadata: {
      sourceType: "video",
      documentTitle: "Pricing Strategy Talk",
      timestampSeconds: 1320,
    },
  }),
  new Document({
    pageContent:
      "Our onboarding flow reduces churn by simplifying the first-week experience.",
    metadata: {
      sourceType: "pdf",
      documentTitle: "Onboarding Report",
      pageNumber: 4,
    },
  }),
  new Document({
    pageContent:
      "Discount bundles were introduced to compete with lower-cost alternatives.",
    metadata: {
      sourceType: "srt",
      documentTitle: "Sales Call Transcript",
      timestampSeconds: 640,
    },
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
  const vectorStore = await MemoryVectorStore.fromDocuments(docs, embedding);
  const selfQueryRetriever = await SelfQueryRetriever.fromLLM({
    llm: llm,
    vectorStore: vectorStore,
    documentContentents:
      "Chunks of lecture/video/PDF content with source metadata",
    attributeInfo: attributeInfo,
    structuredQueryTranslator: new FunctionalTranslator(),
    verbose: true,
  });
  const query =
    "What did they say about pricing after the 20 minute mark in the video?";
  const result = await selfQueryRetriever.invoke(query);
  console.log("Result: ", result);
}
main();
