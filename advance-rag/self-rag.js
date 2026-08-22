import "dotenv/config";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { z } from "zod";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { Document } from "@langchain/core/documents";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

const embedding = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
});
const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
  apiKey: process.env.OPENAI_API_KEY,
});

const rawDocs = [
  new Document({
    pageContent:
      "AlphaCorp announced a new AI chip called QuantumNode scheduled for release in Q4 2026.",
  }),
  new Document({
    pageContent:
      "The QuantumNode chip architecture utilizes 2nm ribbonfet technology and consumes 40% less power.",
  }),
  new Document({
    pageContent:
      "Node.js 26 was released featuring built-in support for compile-to-binary compilation out of the box.",
  }),
];
const vectorStore = await MemoryVectorStore.fromDocuments(rawDocs, embedding);
const baseRetriever = vectorStore.asRetriever({ k: 2 });

// GRADERS
// ==================================================================================
const docGraderSchema = z.object({
  binaryScore: z
    .enum(["yes", "no"])
    .describe("Are the documents relevant to the questions?, 'yes' or 'no'"),
});
const docGrader = llm.withStructuredOutput(docGraderSchema, {
  name: "document_grader",
});
const hellucinationGraderSchema = z.object({
  binaryScore: z
    .enum(["yes", "no"])
    .describe(
      "Is the answer grounded in / supported by the facts provided? 'yes' or 'no'",
    ),
});
const hellucinationGrader = llm.withStructuredOutput(
  hellucinationGraderSchema,
  {
    name: "hellucination_grader",
  },
);
const answerGraderSchema = z.object({
  binaryScore: z
    .enum(["yes", "no"])
    .describe("Does the answer resolve user's questions? 'yes' or 'no'"),
});
const answerGrader = llm.withStructuredOutput(answerGraderSchema, {
  name: "answer_grader",
});
// ==================================================================================

// PROMPTS
// ==================================================================================
const rewritePrompt = ChatPromptTemplate.fromTemplate(
  `You are an AI assistant, please give me a alternate query for the original 
    user query to fetch documents from vector database
    
    Original user query:
    {query}
    `,
);
const rewriteChain = await RunnableSequence.from([
  rewritePrompt,
  llm,
  new StringOutputParser(),
]);
const genPrompt = ChatPromptTemplate.fromTemplate(
  `You are an AI assistant, please answer the given question by only using the context below
    
    Context:
    {context}

    Question:
    {question}

    Answer:
    `,
);
const genPromptChain = await RunnableSequence.from([
  genPrompt,
  llm,
  new StringOutputParser(),
]);
// ==================================================================================

async function runSelfRag(userQuery) {
  let currentQuery = userQuery;
  const MAX_ATTEMPTS = 3;
  let currentAttempt = 0;

  while (currentAttempt < MAX_ATTEMPTS) {
    currentAttempt++;

    const retrievedDocs = await baseRetriever.invoke(userQuery);
    const context = retrievedDocs.map((doc) => doc.pageContent).join("\n");

    const documentGrade = await docGrader.invoke([
      {
        role: "system",
        content:
          "You are a grader assessing relevance of a retrieved document to a user question.",
      },
      {
        role: "human",
        content: `Document Context:\n${context}\n\nUser Question: ${currentQuery}`,
      },
    ]);
    if (documentGrade.binaryScore == "no") {
      currentQuery = await rewritePrompt.invoke({
        query: userQuery,
      });
      continue;
    }

    const answer = await genPromptChain.invoke({
      context: context,
      question: userQuery,
    });

    const hellucationGrade = await hellucinationGrader.invoke([
      {
        role: "system",
        content:
          "Validate if the answer is completely grounded in and factual relative to the context provided.",
      },
      {
        role: "human",
        content: `Context:\n${context}\n\nDraft Answer: ${answer}`,
      },
    ]);
    if (hellucationGrade.binaryScore == "no") {
      currentQuery = await rewritePrompt.invoke({
        query: userQuery,
      });
      continue;
    }

    const answerGrade = await answerGrader.invoke([
      {
        role: "system",
        contents:
          "Assess whether the generated answer fully and directly solves the user's core question.",
      },
      {
        role: "system",
        content: `User Question: ${userQuery}\n\nGenerated Answer: ${answer}`,
      },
    ]);
    if (answerGrade.binaryScore == "yes") {
      console.log("✅ Critique Result: Answer is useful.");
      return { status: "Success", answer };
    } else {
      console.log("❌ Critique Result: Answer is not useful enough.");
    }

    console.log(
      "\n⚠️ Self-RAG loop could not self-correct perfectly. Falling back to direct LLM response.",
    );
    const fallbackAnswer = await llm.invoke(userQuery);
    return { status: "Fallback", answer: fallbackAnswer.content };
  }

  return null;
}

async function main() {
  // Scenario A: Hits the data perfectly
  const res1 = await runSelfRag(
    "What are the specs of AlphaCorp's new AI chip?",
  );
  console.log("\nFINAL RESPONSE 1:", res1.answer);

  const res2 = await runSelfRag(
    "Tell me about the new chip coming out late 2026",
  );
  console.log("\nFINAL RESPONSE 2:", res2.answer);
}
main();
