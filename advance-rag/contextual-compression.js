// contextual-compression-example.js
// Demonstrates ContextualCompressionRetriever wrapping a MemoryVectorStore
// retriever with an LLMChainExtractor compressor.
//
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { LLMChainExtractor } from "@langchain/classic/retrievers/document_compressors/chain_extract";
import { ContextualCompressionRetriever } from "@langchain/classic/retrievers/contextual_compression";
import "dotenv/config";

import OpenAI from "openai";

async function main() {
  // 1. Some sample chunks - imagine these came from your PDF/SRT ingestion pipeline.
  //    Notice each chunk has a mix of relevant + irrelevant sentences, which is
  //    exactly the scenario compression is meant to clean up.
  let docs = [
    new Document({
      pageContent: `A Program Derived Address (PDA) in Solana is an address that is 
        deterministically derived from a program ID and a set of seeds, and
        has no private key. The Solana runtime was first announced in 2017.
        PDAs let programs sign for accounts programmatically, which is core
        to building stateful on-chain logic.",`,
      metadata: { source: "solana-core-concepts.pdf", startTime: null },
    }),
    new Document({
      pageContent: `Rust's ownership model prevents data races at compile time. Token2022
        is an extended SPL token program supporting features like transfer
        fees and confidential transfers. None of this paragraph discusses PDAs.`,
      metadata: { source: "rust-fundamentals.pdf", startTime: null },
    }),
    new Document({
      pageContent: `Geyser plugins stream account and transaction updates out of a
        validator in real time, which is how indexers like Carbon consume
        on-chain data without polling RPC.,`,
      metadata: { source: "yellowstone-grpc-notes.pdf", start_time: null },
    }),
  ];

  const embedding = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
  });
  const llm = new ChatOpenAI({
    model: "gpt-4o-mini",
    apiKey: process.env.OPENAI_API_KEY,
    temperature: 0,
  });

  const vectorStore = await MemoryVectorStore.fromDocuments(docs, embedding);

  // 2. Build the base retriever exactly like you're already doing.
  const baseRetriever = vectorStore.asRetriever({ k: 2 });

  const query = "What is a PDA in Solana?";

  // --- Before compression: raw chunks, full noise included ---
  const rawDocs = await baseRetriever.invoke(query);
  console.log("=== BEFORE compression ===");
  rawDocs.forEach((d, i) =>
    console.log(`[${i}] (${d.pageContent.length} chars) ${d.pageContent}\n`),
  );

  // 3. Build the compressor. LLMChainExtractor uses an LLM call per doc to pull
  //    out only the query-relevant statements.
  const compressor = LLMChainExtractor.fromLLM(llm);
  const compressionRetriever = new ContextualCompressionRetriever({
    baseRetriever,
    baseCompressor: compressor,
  });
  // --- After compression: only the relevant statements survive ---
  const compressedDocs = await compressionRetriever.invoke(query);
  console.log("=== AFTER compression ===");
  compressedDocs.forEach((d, i) =>
    console.log(`[${i}] (${d.pageContent.length} chars) ${d.pageContent}\n`),
  );

  // Rough token/character savings, useful when you're deciding whether the
  // extra LLM call per doc is worth it for your use case.
  const beforeLen = rawDocs.reduce((n, d) => n + d.pageContent.length, 0);
  const afterLen = compressedDocs.reduce((n, d) => n + d.pageContent.length, 0);
  console.log(
    `Compression ratio: ${(beforeLen / afterLen).toFixed(2)}x smaller`,
  );
}

main().catch(console.error);
