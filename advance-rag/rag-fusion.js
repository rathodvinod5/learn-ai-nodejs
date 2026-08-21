import "dotenv/config";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { Document } from "@langchain/core/documents";

const rawDocs = [
  new Document({
    pageContent:
      "The architectural design of the Eiffel Tower was created by Maurice Koechlin and Emile Nouguier.",
    metadata: { id: 1 },
  }),
  new Document({
    pageContent:
      "The Eiffel Tower was constructed from 1887 to 1889 as the entrance to the 1889 World's Fair.",
    metadata: { id: 2 },
  }),
  new Document({
    pageContent:
      "Gustave Eiffel's company built the famous Parisian tower, but he didn't design the structure himself.",
    metadata: { id: 3 },
  }),
  new Document({
    pageContent:
      "The Louvre Museum is the world's largest art museum and a historic monument in Paris.",
    metadata: { id: 4 },
  }),
];

const embedding = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
});
const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
  apiKey: process.env.OPENAI_API_KEY,
});
const vectorStore = await MemoryVectorStore.fromDocuments(rawDocs, embedding);

async function generateMultipleQueries(userQuery) {
  const multiQueryPrompt = ChatPromptTemplate.fromTemplate(
    `You are an AI assistant. Generate 3 alternative versions of the 
    following user query to retrieve relevant documents from a vector database.
    Provide these alternative queries separated by newlines. 
    
    Original query:
    {query}`,
  );
  const multiQueryChat = await RunnableSequence.from([
    multiQueryPrompt,
    llm,
    new StringOutputParser(),
  ]);
  let generatedQueries = await multiQueryChat.invoke({
    query: userQuery,
  });
  generatedQueries = generatedQueries
    .split("\n")
    .map((line) => line.replace(/^\d+\.\s*/, "").trim())
    .filter(Boolean);
  return [userQuery, ...generatedQueries];
}

async function reciprocalRankFusion(vectorDocs, k = 60) {
  const rrScores = {};
  const docMapping = {};

  vectorDocs.map((docs) => {
    docs.map((doc, rank) => {
      const contents = doc.pageContent;
      docMapping[contents] = doc;

      if (!rrScores[contents]) rrScores[contents] = 0;

      rrScores[contents] += 1 / (k + (rank + 1));
    });
  });

  const reRankedDocs = Object.keys(rrScores)
    .sort((a, b) => rrScores[b] - rrScores[a])
    .map((content) => docMapping[content]);

  // console.log(rrScores, "\n", reRankedDocs);
  return reRankedDocs;
}

async function runRAGFusion() {
  const userQuery = "Who designed the Eiffel Tower?";
  const generatedQueries = await generateMultipleQueries(userQuery);

  const promisedQueries = generatedQueries.map((query) =>
    vectorStore.similaritySearch(query, 2),
  );
  let vectorDocs = await Promise.all(promisedQueries);
  const rerankedDocs = await reciprocalRankFusion(vectorDocs);

  const context = rerankedDocs
    .slice(0, 2)
    .map((doc) => doc.pageContent)
    .join("\n");

  const finalPrompt = ChatPromptTemplate.fromTemplate(
    `Please answer the user query by only using the context below
    
    Context:
    {context}
    
    User query:
    {query}`,
  );
  const finalChain = RunnableSequence.from([
    finalPrompt,
    llm,
    new StringOutputParser(),
  ]);
  const finalResult = await finalChain.invoke({
    context: context,
    query: userQuery,
  });
  console.log("finalResult: ", finalResult);
}
runRAGFusion().catch((err) => console.log("ERR: ", err));
