import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import "dotenv/config";

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
  new Document({
    pageContent: `Chapter 11: VPN Client Setup. To connect via VPN, install the client,
    import the .ovpn profile, and authenticate with SSO. VPN traffic uses
    UDP port 1194 by default; if your firewall blocks that range, switch
    the profile to TCP port 443 instead.`,
    metadata: { source: "network-manual.pdf", chapter: 11 },
  }),
];

async function main() {
  // ---------------------------------------------------------------
  // 1. Source documents + base retriever. Deliberately retrieved WIDE
  //    (k: 6) here — stage 1 is supposed to be a loose net; precision
  //    is stage 2's job, not this one's.
  // ---------------------------------------------------------------
  const embedding = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
  });
  const llm = new ChatOpenAI({
    model: "gpt-4o-mini",
    temperature: 0,
    apiKey: process.env.OPENAI_API_KEY,
  });
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 300,
    chunkOverlap: 30,
  });
  const splitDocs = await splitter.splitDocuments(rawDocs);
  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embedding,
  );
  const baseRetiever = vectorStore.asRetriever({ k: 6 });
  const question = "How do I fix DNS lookups timing out because of a firewall?";

  // ---------------------------------------------------------------
  // 2. Stage 1 — vector retrieval. This is the bi-encoder pass: query
  //    and chunks were embedded independently, so ranking here is only
  //    an approximation of true relevance.
  // ---------------------------------------------------------------
  const candidates = await baseRetiever.invoke(question);
  console.log(`Question: "${question}"\n`);
  console.log(
    `Stage 1 — vector retrieval (${candidates.length} candidates, unranked by true relevance):`,
  );
  candidates.forEach((doc, i) => {
    console.log(
      `  [${i}] (${doc.metadata.source}${doc.metadata.chapter ? `, ch.${doc.metadata.chapter}` : ""}) ${doc.pageContent.slice(0, 90)}...`,
    );
  });
  console.log();

  // ---------------------------------------------------------------
  // 3. Stage 2 — LLM re-ranking. The query and every candidate are
  //    passed into ONE prompt so the model scores them jointly, the
  //    way a cross-encoder would. In production you'd more likely
  //    reach for a dedicated cross-encoder (e.g. Cohere Rerank,
  //    bge-reranker) — an LLM call is heavier, but the mechanics
  //    are identical and easier to inspect here.
  // ---------------------------------------------------------------
  const RERANK_TOP_N = 2;
  const rerankPrompt = ChatPromptTemplate.fromTemplate(
    `You are a relevance-scoring engine. Given the QUESTION and a list of
    CANDIDATE passages, score each passage 0-10 on how directly and
    specifically it helps answer the QUESTION (10 = answers it precisely,
    0 = irrelevant). Judge meaning, not keyword overlap.

    QUESTION:
    {question}

    CANDIDATES:
    {candidates}

    Return ONLY a JSON array like [{{"index":0,"score":7}}], no other text.
    `,
  );
  const rerankChain = await RunnableSequence.from([
    rerankPrompt,
    llm,
    new StringOutputParser(),
  ]);
  const candidatesBlock = candidates
    .map((doc, i) => `[${i}] ${doc.pageContent.replace(/\s+/g, " ")}`)
    .join("\n");
  const rawScores = await rerankChain.invoke({
    question,
    candidates: candidatesBlock,
  });
  const cleaned = rawScores
    .trim()
    .replace(/```json|```/g, "")
    .trim();
  const scores = JSON.parse(cleaned);
  console.log("scores: ", scores);
  const reranked = scores
    .map(({ index, score }) => ({ doc: candidates[index], score }))
    .sort((a, b) => b.score - a.score)
    .slice(0, RERANK_TOP_N);

  console.log(
    `Stage 2 — LLM re-ranking (top ${RERANK_TOP_N} by true relevance):`,
  );
  reranked.forEach(({ doc, score }, i) => {
    console.log(
      `  ${i + 1}. [score ${score}] (${doc.metadata.source}${doc.metadata.chapter ? `, ch.${doc.metadata.chapter}` : ""}) ${doc.pageContent.slice(0, 90)}...`,
    );
  });
  console.log();

  // ---------------------------------------------------------------
  // 4. Generate the final answer from ONLY the re-ranked context —
  //    this is the payoff: the LLM never sees the candidates that
  //    stage 1 pulled in but stage 2 correctly downgraded.
  // ---------------------------------------------------------------
  const finalPrompt = ChatPromptTemplate.fromTemplate(
    `Please answer the questions only from the context given below

    Context:
    {context}

    Question:
    {question}

    Answer:
    `,
  );
  const finalChain = await RunnableSequence.from([
    finalPrompt,
    llm,
    new StringOutputParser(),
  ]);
  const result = await finalChain.invoke({
    context: candidates.map((item) => item.pageContent).join("\n\n"),
    question: question,
  });
  console.log("=== Final Answer ===");
  console.log(result);
}
main();
