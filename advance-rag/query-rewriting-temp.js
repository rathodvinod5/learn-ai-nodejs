import "dotenv/config";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableLambda, RunnableSequence } from "@langchain/core/runnables";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

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
    metadata: { source: "network-manula.pdf", chapter: 9 },
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
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 300,
    chunkOverlap: 30,
  });
  const splitDocs = await splitter.splitDocuments(rawDocs);
  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embedding,
  );
  const baseRetriever = vectorStore.asRetriever({ k: 3 });
  const rewritePrompt = ChatPromptTemplate.fromTemplate(
    `Given the conversation history and a follow-up question, rewrite the
    follow-up into a standalone question that contains all necessary context.
    Do not answer it — only rewrite it.

    Chat History:
    {history}

    Follow-up question: {question}

    Standalone question:`,
  );
  const chatHistory = [
    "User: I keep getting DNS timeouts on our network.",
    "Assistant: That's usually caused by a firewall blocking outbound UDP on port 53.",
  ].join("\n");

  const followUpQuestion = "what about that port issue you mentioned?";

  const rewriteChain = await RunnableSequence.from([
    rewritePrompt,
    llm,
    new StringOutputParser(),
  ]);
  const standaloneQuestion = await rewriteChain.invoke({
    history: chatHistory,
    question: followUpQuestion,
  });
  console.log(`Follow up question: `, followUpQuestion);
  console.log(`Standalone Question: `, standaloneQuestion);

  const NUM_QUERIES = 3;
  const multiQueryPrompt = ChatPromptTemplate.fromTemplate(
    `You are an assistant that generates multiple search queries based on a 
    single input questions, to improve document retrieval coverage.

    Generate exactly ${NUM_QUERIES} different versions of the question below 
    without any numbering, bullet points or extra commentary.

    Question:
    {question}
    `,
  );
  const parsedLines = new RunnableLambda({
    func: (text) =>
      text
        .split("\n")
        .map((line) => line.trim())
        .filter((line) => line.length > 0),
  });
  const multiQueryChain = await RunnableSequence.from([
    multiQueryPrompt,
    llm,
    new StringOutputParser(),
    parsedLines,
  ]);
  const queryVariants = await multiQueryChain.invoke({
    question: parsedLines,
  });
  console.log("queryVariants: ", queryVariants);

  const fetchAndMerge = new RunnableLambda({
    func: async (queries) => {
      const seen = new Set();
      const merged = [];
      const result = await Promise.all(
        queries.map((q) => baseRetriever.invoke(q)),
      );

      for (const docs of result) {
        for (const doc of docs) {
          const { pageContent } = doc;
          if (!seen.has(pageContent)) {
            seen.add(pageContent);
            merged.push(pageContent);
          }
        }
      }

      return merged;
    },
  });
  const retrievedDocs = await fetchAndMerge.invoke(queryVariants);

  const finalPrompt = ChatPromptTemplate.fromTemplate(
    `Answer the question from the context given below
    
    Context:
    {context}

    Question:
    {question}

    Answer:
    `,
  );
  const finalChain = RunnableSequence.from([
    finalPrompt,
    llm,
    new StringOutputParser(),
  ]);
  const result = await finalChain.invoke({
    context: retrievedDocs.map((docs) => docs.pageContent).join("\n\n"),
    question: standaloneQuestion,
  });
  console.log("\nResult: ", result);
}
main();
