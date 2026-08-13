/**
 "Multi-query" and "Query rewriting" they're closely related 
 — multi-query is essentially a specific application of query rewriting, 
 but there's a meaningful distinction worth understanding, especially since 
 you're implementing both directions of this in your RAG pipeline.
 
* Query Rewriting in RAG — Working Example
 * -----------------------------------------------------------------
 * Two techniques demonstrated together, both feeding a shared retriever:
 *
 * 1. CONVERSATIONAL REWRITING
 *    Turns a context-dependent follow-up ("what about the port issue?")
 *    into a standalone question using chat history, via an LLM call.
 *
 * 2. MULTI-QUERY REWRITING
 *    Takes that standalone question and asks the LLM to generate several
 *    paraphrased variants. Each variant is embedded and searched
 *    separately; the union of results is deduped and returned. This
 *    covers vocabulary mismatches a single query would miss.
 *
 * Pipeline: chat history + follow-up -> [rewrite] -> standalone question
 *           -> [multi-query expand] -> N search queries -> retrieved docs
 *
 * Dependencies (LangChain.js v1):
 *   npm install @langchain/classic @langchain/openai @langchain/core
 *
 * Env:
 *   export OPENAI_API_KEY=sk-...
 */

import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "@langchain/classic/text_splitter";
import { Document } from "@langchain/core/documents";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence, RunnableLambda } from "@langchain/core/runnables";

async function main() {
  // ---------------------------------------------------------------
  // 1. Source documents + base retriever (plain vector search here;
  //    swap in the ParentDocumentRetriever from the earlier example
  //    if you want parent-chunking + query rewriting combined).
  // ---------------------------------------------------------------
  const rawDocs = [
    new Document({
      pageContent: `Pricing: The Starter plan is $9/month and includes 5 projects. The Pro
        plan is $29/month and includes unlimited projects plus priority support.
        The Enterprise plan is custom-priced and includes SSO and a dedicated
        account manager.`,
      metadata: { source: "pricing.md" },
    }),
    new Document({
      pageContent: `Chapter 4: DNS Resolution Failures. Symptom: DNS lookups time out
        intermittently. Root cause: a firewall silently blocking outbound UDP
        traffic on port 53. Fix: open port 53 or switch to DNS-over-HTTPS, then
        verify with 'dig @resolver example.com'.`,
      metadata: { source: "network-manual.pdf", chapter: 4 },
    }),
    new Document({
      pageContent: `Chapter 9: Firewall Port Configuration. Ports 80 and 443 must be open
        for standard web traffic. Port 22 should be restricted to known IPs for
        SSH. Blocking outbound UDP/53 without an alternate DNS path is a common
        misconfiguration that breaks name resolution app-wide.`,
      metadata: { source: "network-manual.pdf", chapter: 9 },
    }),
  ];

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 300,
    chunkOverlap: 30,
  });
  const splitDocs = await splitter.splitDocuments(rawDocs);

  const vectorstore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    new OpenAIEmbeddings(),
  );
  const baseRetriever = vectorstore.asRetriever({ k: 3 });

  const llm = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });

  // ---------------------------------------------------------------
  // 2. Conversational rewriting: collapse history + follow-up into
  //    one standalone question. This is the step every chat-style
  //    RAG app needs, since raw follow-ups are unembeddable on their own.
  // ---------------------------------------------------------------
  const rewritePrompt = ChatPromptTemplate.fromTemplate(
    `Given the conversation history and a follow-up question, rewrite the
    follow-up into a standalone question that contains all necessary context.
    Do not answer it — only rewrite it.

    Chat History:
    {history}

    Follow-up question: {question}

    Standalone question:`,
  );

  const rewriteChain = RunnableSequence.from([
    rewritePrompt,
    llm,
    new StringOutputParser(),
  ]);

  const chatHistory = [
    "User: I keep getting DNS timeouts on our network.",
    "Assistant: That's usually caused by a firewall blocking outbound UDP on port 53.",
  ].join("\n");

  const followUp = "what about that port issue you mentioned?";

  const standaloneQuestion = await rewriteChain.invoke({
    history: chatHistory,
    question: followUp,
  });

  console.log(`Original follow-up: "${followUp}"`);
  console.log(
    `Rewritten standalone question: "${standaloneQuestion.trim()}"\n`,
  );

  // ---------------------------------------------------------------
  // 3. Multi-query rewriting — built with plain LCEL instead of the
  //    legacy MultiQueryRetriever class (which wraps the deprecated
  //    LLMChain under the hood). This is the current idiomatic
  //    pattern: prompt -> LLM -> parse into N queries -> fan out to
  //    the retriever in parallel -> merge + dedupe by content.
  // ---------------------------------------------------------------
  const QUERY_COUNT = 3;

  const multiQueryPrompt = ChatPromptTemplate.fromTemplate(
    `You are an assistant that generates multiple search queries based on a
    single input question, to improve document retrieval coverage.

    Generate exactly ${QUERY_COUNT} different versions of the question below,
    each on its own line, with no numbering, bullets, or extra commentary.

    Question: {question}`,
  );

  // Turns the raw newline-separated LLM output into a clean string[].
  const parseLines = new RunnableLambda({
    func: (text) =>
      text
        .split("\n")
        .map((line) => line.trim())
        .filter((line) => line.length > 0),
  });

  const generateQueries = RunnableSequence.from([
    multiQueryPrompt,
    llm,
    new StringOutputParser(),
    parseLines,
  ]);

  // Fans a list of queries out to the retriever in parallel, then
  // merges results and dedupes by page content (a doc matched by
  // two different query variants should only appear once).
  const fetchAndMerge = new RunnableLambda({
    func: async (queries) => {
      const resultsPerQuery = await Promise.all(
        queries.map((q) => baseRetriever.invoke(q)),
      );
      const seen = new Set();
      const merged = [];
      for (const docs of resultsPerQuery) {
        for (const doc of docs) {
          if (!seen.has(doc.pageContent)) {
            seen.add(doc.pageContent);
            merged.push(doc);
          }
        }
      }
      return merged;
    },
  });

  // (generateQueries and fetchAndMerge are invoked as two steps below so we
  // can log the generated query variants; in production you'd likely chain
  // them as one: RunnableSequence.from([generateQueries, fetchAndMerge]))

  const queryVariants = await generateQueries.invoke({
    question: standaloneQuestion,
  });
  console.log(`Generated ${queryVariants.length} query variant(s):`);
  queryVariants.forEach((q, i) => console.log(`  ${i + 1}. ${q}`));
  console.log();

  const retrievedDocs = await fetchAndMerge.invoke(queryVariants);

  console.log(
    `Retrieved ${retrievedDocs.length} unique chunk(s) across all query variants:\n`,
  );
  retrievedDocs.forEach((doc, i) => {
    console.log(
      `--- Chunk ${i + 1} (${doc.metadata.source}${doc.metadata.chapter ? `, ch.${doc.metadata.chapter}` : ""}) ---`,
    );
    console.log(doc.pageContent.slice(0, 120) + "...\n");
  });

  // ---------------------------------------------------------------
  // 4. Generate the final answer from the retrieved context.
  // ---------------------------------------------------------------
  const answerPrompt = ChatPromptTemplate.fromTemplate(
    `Answer the question using only the context below.

    Context:
    {context}

    Question: {question}

    Answer:`,
  );

  const answerChain = RunnableSequence.from([
    answerPrompt,
    llm,
    new StringOutputParser(),
  ]);

  const answer = await answerChain.invoke({
    context: retrievedDocs.map((d) => d.pageContent).join("\n\n"),
    question: standaloneQuestion,
  });

  console.log("=== Final Answer ===");
  console.log(answer);
}

main().catch(console.error);
