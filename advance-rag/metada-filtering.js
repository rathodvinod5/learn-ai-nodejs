import "dotenv/config";

import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
// import { MemoryVectorStore } from "langchain/core/vectorstores/memory";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createRetrievalChain } from "@langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";

async function init(q) {
  // 1. Setup and Ingestion
  const embeddingModel = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
    model: "text-embedding-3-small",
  });

  const documents = [
    new Document({
      pageContent: "Node.js is an open-source JavaScript runtime environment.",
      metadata: {
        category: "technology",
        year: 2015,
        status: "public",
      },
    }),
    new Document({
      pageContent: "Python is an open-source JavaScript runtime environment.",
      metadata: {
        category: "technology",
        year: 2005,
        status: "public",
      },
    }),
    new Document({
      pageContent:
        "The internal strategy document outlines our Q3 financial goals",
      metadata: {
        category: "finance",
        year: 2010,
        status: "private",
      },
    }),
  ];

  const vectorStore = await MemoryVectorStore.fromDocuments(
    documents,
    embeddingModel,
  );

  const query = q ?? "";

  // 2. Querying with Metadata Filters
  const filter = (doc) =>
    doc.metadata.status === "public" && doc.metadata.year == 2005;
  const runtimeResults = await vectorStore.similaritySearch(query, 1, filter);
  console.log(JSON.stringify(runtimeResults[0], null, 2));

  // // 3. Integrating into a LangChain Retrieval Chain
  // const llm = new ChatOpenAI({
  //   apiKey: process.env.OPENAI_API_KEY,
  //   model: "gpt-4o",
  // });

  // const retriever = vectorStore.asRetriever({
  //   filter: filter,
  //   k: 2,
  // });

  // const chatPromptTemplate = ChatPromptTemplate.fromMessages([
  //   ["system", `Answer the questions based only on the context: \n\n{context}`],
  //   ["human", `{input}`],
  // ]);

  // const combineDocsChain = await createStuffDocumentsChain({
  //   llm,
  //   prompt: questionAnsweringPrompt,
  // });

  // const ragChain = await createRetrievalChain({
  //   retriever,
  //   combineDocsChain,
  // });

  // const response = await ragChain.invoke({
  //   input: query,
  // });

  // console.log(response.answer);
}

// init("Tell me about runtime environments");
init("What is python and why it is used for");
