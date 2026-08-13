import "dotenv/config";

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { OpenAIEmbeddings } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";

async function generateVectoreEmbeddingForFile(filepath) {
  const loader = new PDFLoader(filepath);
  const document = await loader.load();

  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
    apiKey: process.env.OPENAI_API_KEY,
  });

  const vectorStore = await QdrantVectorStore.fromExistingCollection(
    embeddings,
    {
      url: "http://localhost:6333",
      collectionName: "chai-code-rag-tutorial",
    },
  );

  await vectorStore.addDocuments(document);
  console.log("docs stored");
}

generateVectoreEmbeddingForFile("software.pdf");
