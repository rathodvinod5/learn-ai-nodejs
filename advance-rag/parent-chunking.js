import "dotenv/config";
/**
 * Parent Document Retrieval (Parent Chunking) — Working Example
 * -----------------------------------------------------------------
 * Concept:
 *   - Small "child" chunks are embedded and searched for precision.
 *   - Large "parent" chunks are stored separately (not embedded) and
 *     returned to the LLM once a child match is found, so generation
 *     has full context instead of a tiny isolated snippet.
 *
 * Dependencies (LangChain.js v1):
 *   npm install @langchain/classic @langchain/openai @langchain/core
 *
 * Env:
 *   export OPENAI_API_KEY=sk-...
 */

// NOTE: As of LangChain.js v1.0, legacy pieces like ParentDocumentRetriever
// were moved out of the slimmed-down `langchain` package into `@langchain/classic`.
// npm install @langchain/classic @langchain/openai @langchain/core
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/classic/text_splitter";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { InMemoryStore } from "@langchain/core/stores";
import { ParentDocumentRetriever } from "@langchain/classic/retrievers/parent_document";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";

// import { InMemoryStore } from "@langchain/classic/storage/in_memory";

async function main() {
  // ---------------------------------------------------------------
  // 1. Source documents (imagine these came from your PDF/SRT/audio
  //    ingestion pipeline — each Document is one "source unit").
  // ---------------------------------------------------------------
  const rawDocs = [
    new Document({
      pageContent: `Chapter 4: DNS Resolution Failures

Symptom: Users report that DNS lookups intermittently time out, especially
during peak traffic hours. The issue is inconsistent — some queries resolve
fine while others hang for 5-10 seconds before failing.

Root Causes: The most common cause is a corporate firewall silently
dropping outbound UDP traffic on port 53, forcing clients to fall back to
slower TCP retries. A second common cause is a misconfigured resolv.conf
pointing to a resolver that is no longer reachable. Less commonly, the
upstream DNS provider itself may be experiencing a partial outage.

Fix: First verify basic connectivity with 'dig @<resolver> example.com'.
If UDP/53 is blocked, either open the port or configure the client to use
DNS-over-TCP or DNS-over-HTTPS. If resolv.conf is stale, update it to
point to a verified working resolver (e.g. 1.1.1.1 or 8.8.8.8) and flush
the local DNS cache. If the upstream provider is down, switch to a backup
resolver until service is restored.`,
      metadata: { source: "network-manual.pdf", chapter: 4 },
    }),
    new Document({
      pageContent: `Chapter 7: Slow Wi-Fi Performance

Symptom: Wi-Fi throughput drops significantly in the afternoon on the
office network, even though signal strength appears normal.

Root Causes: Channel congestion from neighboring access points is the
leading cause, particularly on the crowded 2.4GHz band. Outdated access
point firmware and too many clients on a single AP are secondary causes.

Fix: Run a spectrum scan to identify a cleaner channel, migrate clients to
the 5GHz or 6GHz band where possible, and update AP firmware. If a single
AP is overloaded, add a second AP and enable band steering.`,
      metadata: { source: "network-manual.pdf", chapter: 7 },
    }),
  ];

  // ---------------------------------------------------------------
  // 2. Splitters
  //    - parentSplitter: keeps parent chunks reasonably sized if a
  //      source document is huge (optional — you can skip this and
  //      use whole documents as parents).
  //    - childSplitter: produces the small, embedding-friendly pieces.
  // ---------------------------------------------------------------
  const parentSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 2000,
    chunkOverlap: 0,
  });
  const childSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 300,
    chunkOverlap: 30,
  });

  // ---------------------------------------------------------------
  // 3. Stores
  //    - vectorstore: holds ONLY child chunk embeddings.
  //    - docstore: plain key/value store holding full parent chunks,
  //      keyed by an auto-generated parent id. In production swap
  //      InMemoryStore for Redis/S3/Postgres-backed storage.
  // ---------------------------------------------------------------
  const embedding = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
  });
  const vectorStore = new MemoryVectorStore(embedding);
  const docStore = new InMemoryStore();

  const retriever = new ParentDocumentRetriever({
    vectorstore: vectorStore,
    byteStore: docStore,
    parentSplitter: parentSplitter,
    childSplitter: childSplitter,
    parentK: 2,
    childK: 4,
  });

  // ---------------------------------------------------------------
  // 4. Ingest — this automatically:
  //      a) splits each doc into parent chunks
  //      b) splits each parent into child chunks
  //      c) embeds + stores children in vectorstore (with parent_id metadata)
  //      d) stores parents in docstore
  // ---------------------------------------------------------------
  await retriever.addDocuments(rawDocs);

  // ---------------------------------------------------------------
  // 5. Query — retrieval happens against small child chunks, but
  //    what comes back is the full parent chunk(s).
  // ---------------------------------------------------------------
  const userQuery = "My DNS lookups keep timing out, what should I check?";
  const retrievedParents = await retriever.invoke(userQuery);
  console.log(`\nQuery: "${userQuery}"`);
  console.log(`\nRetrieved ${retrievedParents.length} parent chunk(s):\n`);
  retrievedParents.forEach((doc, i) => {
    console.log(
      `--- Parent ${i + 1} (source: ${doc.metadata.source}, chapter: ${doc.metadata.chapter}) ---`,
    );
    console.log(doc.pageContent.slice(0, 150) + "...\n");
  });

  // ---------------------------------------------------------------
  // 6. Feed the parent chunk(s) to the LLM for generation
  // ---------------------------------------------------------------
  const llm = new ChatOpenAI({
    model: "gpt-4o-mini",
    temperature: 0,
  });
  const template = ChatPromptTemplate.fromTemplate(
    `Answer the question by using only the context below, cite the chapter number.
        
    Context:
    {context}

    Question:
    {question}

    Answer:`,
  );
  const chain = RunnableSequence.from([
    {
      context: (input) =>
        input.docs
          .map((d) => `[Chapter ${d.metadata.chapter}]\n${d.pageContent}`)
          .join("\n\n"),
      question: (input) => input.question,
    },
    template,
    llm,
    new StringOutputParser(),
  ]);

  const answer = await chain.invoke({
    docs: retrievedParents,
    question: userQuery,
  });

  console.log("=== LLM Answer ===");
  console.log(answer);
}

main().catch(console.error);
