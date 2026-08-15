import "dotenv/config";

import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { Document } from "@langchain/core/documents";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";

async function createSentenceWindowDocs(text, windowSize) {
  const cleanedText = text.replace(/\s+/g, " ");
  const sentences = cleanedText.match(/[^.!?]+[.!?]+(\s|$)/g) || [cleanedText];
  const docs = [];

  for (let index = 0; index < sentences.length; index++) {
    const currSentence = sentences[index].trim();
    if (!currSentence) continue;

    const start = Math.max(0, index - windowSize);
    const end = Math.min(sentences.length, index - windowSize + 1);
    const context = sentences.slice(start, end);

    docs.push(
      new Document({
        pageContent: currSentence,
        metadata: {
          windowContext: context,
          sentenceIndex: index,
        },
      }),
    );
  }
  return docs;
}

function metadataReplacementNode(docs) {
  return docs
    .map((doc) => {
      console.log(`\n🎯 Search Match: "${doc.pageContent}"`);
      console.log(
        `🔄 Window Expanded Context: "${doc.metadata.windowContext}"`,
      );
      return doc.metadata.windowContext;
    })
    .join("\n\n");
}

async function main() {
  const WINDOW_SIZE = 2;
  const rawText = `
    The company remote work policy was updated for 2026. Employees may work from anywhere within the country. 
    However, everyone must core-align with their team hours between 10 AM and 2 PM EST. 
    A monthly stipend of $150 is provided to cover home internet and utilities. 
    Expense reports for this stipend must be submitted by the 25th of each month. 
    Late submissions will result in delayed payouts until the following cycle.
  `;

  const embedding = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
  });
  const llm = new ChatOpenAI({
    model: "gpt-4o-mini",
    apiKey: process.env.OPENAI_API_KEY,
    temperature: 0,
  });
  const documents = await createSentenceWindowDocs(rawText, WINDOW_SIZE);
  const vectorStore = await MemoryVectorStore.fromDocuments(
    documents,
    embedding,
  );
  const baseRetriever = await vectorStore.asRetriever({ k: 2 });
  const prompt = await ChatPromptTemplate.fromTemplate(
    `You are an helpfull AI assitent, answer the question by using only the context provided below
    
    Context:
    {context}

    Question:
    {question}
    `,
  );
  const promtChain = await RunnableSequence.from([
    prompt,
    llm,
    new StringOutputParser(),
  ]);
  const query =
    "What happens if I submit my internet bill expense on the 26th?";
  const retrievedDocsFromVectorDb = await baseRetriever.invoke(query);
  const context = metadataReplacementNode(retrievedDocsFromVectorDb);
  const result = await promtChain.invoke({
    context: context,
    question: query,
  });
  console.log("result: ", result);
}
main();
