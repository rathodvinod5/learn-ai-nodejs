import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import "dotenv/config";

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
    `You are an AI assistence, please generate 3 alternate versions of given 
    user query to fetch documnets from the vector database.

    Prove these alternative queries seperated by newline, without any numbering as well.
    
    Original user query:
    {query}`,
  );
  const multiQueryChain = RunnableSequence.from([
    multiQueryPrompt,
    llm,
    new StringOutputParser(),
  ]);
  let alternateQueries = await multiQueryChain.invoke({
    query: userQuery,
  });
  alternateQueries = alternateQueries
    .split("\n")
    .map((lines) => lines.trim())
    .filter(Boolean);

  return [userQuery, ...alternateQueries];
}

async function reciprocalRankFusion(vectorDocs, k = 60) {
  const rrScore = {};
  const docMapping = {};

  vectorDocs.map((docs) => {
    docs.map((doc, rank) => {
      const contents = doc.pageContent;
      docMapping[contents] = doc;

      if (!rrScore[contents]) rrScore[contents] = 0;

      rrScore[contents] += 1 / (k + (rank + 1));
    });
  });
  const reRankedDocs = Object.keys(rrScore)
    .sort((a, b) => rrScore[b] - rrScore[a])
    .map((contents) => docMapping[contents]);
  return reRankedDocs;
}

async function runRAGFusion() {
  const userQuery = "Who designed the Eiffel Tower?";
  const alternateQueries = await generateMultipleQueries(userQuery);

  let vectorDocs = alternateQueries.map((docs) =>
    vectorStore.similaritySearch(docs, 2),
  );
  vectorDocs = await Promise.all(vectorDocs);

  const reRankedDocs = await reciprocalRankFusion(vectorDocs);
  const context = reRankedDocs
    .slice(0, 2)
    .map((docs) => docs.pageContent)
    .join("\n");

  const finalPrompt = ChatPromptTemplate.fromTemplate(
    `You are an AI assitent, Please answer the user query by using only the context given below.
    
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
  const result = await finalChain.invoke({
    context: context,
    query: userQuery,
  });
  console.log("result: ", result);
}
runRAGFusion().catch((err) => console.log("ERR: ", err));

// Flow diagram
// [ User Query ]
//        │
//        ▼
// ┌────────────────────────────────────────┐
// │ OpenAI Query Expansion                 │ (Generates 3-4 variations)
// └────────────────────────────────────────┘
//        │
//    ┌───┼───┐
//    ▼   ▼   ▼
//  [Q1] [Q2] [Q3]  (Parallel Vector Search)
//    │   │   │
//    ▼   ▼   ▼
//  [D1] [D2] [D3]  (Raw Document Results with original ranks)
//    │   │   │
//    └───┼───┘
//        ▼
// ┌────────────────────────────────────────┐
// │ Reciprocal Rank Fusion (RRF) Algorithm │ (Calculates new scores based on rank)
// └────────────────────────────────────────┘
//        │
//        ▼
// ┌────────────────────────────────────────┐
// │ Top-K Re-ranked Documents              │ (Highest scoring context chunks)
// └────────────────────────────────────────┘
//        │
//        ▼
// ┌────────────────────────────────────────┐
// │ OpenAI Final Generation                │ (Synthesises answer using context)
// └────────────────────────────────────────┘
//        │
//        ▼
// [ Final Output ]
