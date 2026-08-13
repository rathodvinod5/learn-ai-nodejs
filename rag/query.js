import "dotenv/config";

import { OpenAIEmbeddings } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { OpenAI } from "openai";

async function query(userQuery) {
  // convert user query to vector embeddings
  const embedding = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
    apiKey: process.env.OPENAI_API_KEY,
  });

  // search the vector in the qdrant(similiar search)
  const vectorStore = await QdrantVectorStore.fromExistingCollection(
    embedding,
    {
      url: "http://localhost:6333",
      collectionName: "chai-code-rag-tutorial",
    },
  );

  // get similiar vector and chunks
  const vectoreRetriever = vectorStore.asRetriever({ k: 5 });
  const result = await vectoreRetriever.invoke(userQuery);

  const SYSTEM_PROMPT = `
    You are an expert in answering user query based on the provided context about document.
    Do not answer anything beyond what is not provided

    Always answer the user in short and tell on which page number that content is available and also name of the book.

    User Documents:
    ${result
      .map((e) =>
        JSON.stringify({
          bookName: e.metadata.source,
          pageContent: e.pageContent,
          pageNumber: e.metadata.loc.pageNumber,
        }),
      )
      .join("\n\n")}
  `;

  // feed those chunks to the llm along with the user query
  const client = new OpenAI();
  let llmResponse = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: userQuery },
    ],
  });

  console.log("RESPONSE: ", llmResponse.choices[0].message.content);
}

// query("What is waterfal model");
query("What is black white box and black box testing");
