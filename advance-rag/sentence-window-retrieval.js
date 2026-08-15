import "dotenv/config";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";

async function createSentenceWindowDocs(rawText, WINDOW_SIZE) {
  const cleanedText = rawText.replace(/\s+/g, " ");
  const sentences = cleanedText.match(/[^.!?]+[.!?]+(\s|$)/g) || [cleanedText];
  const documents = [];

  for (let index = 0; index < sentences.length; index++) {
    const currSentence = sentences[index].trim();
    if (!currSentence) continue;

    const start = Math.max(0, index - WINDOW_SIZE);
    const end = Math.min(sentences.length, index + WINDOW_SIZE + 1);
    const context = sentences.slice(start, end);

    documents.push(
      new Document({
        pageContent: currSentence,
        metadata: {
          windowContext: context,
          sentenceIndex: index,
        },
      }),
    );
  }

  return documents;
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
  const baseRetriever = vectorStore.asRetriever({ k: 2 });
  const prompt = ChatPromptTemplate.fromTemplate(
    `You are an helpfull AI assistant, please answer the question by only using the below context.
        
    Context:
    {context}

    Question:
    {question}
    `,
  );
  const promptChain = await RunnableSequence.from([
    prompt,
    llm,
    new StringOutputParser(),
  ]);
  const query =
    "What happens if I submit my internet bill expense on the 26th?";
  const retrievedQuery = await baseRetriever.invoke(query);
  const queryWithContext = metadataReplacementNode(retrievedQuery);
  const result = await promptChain.invoke({
    context: queryWithContext,
    question: query,
  });
  console.log("Result: ", result);
}
main();
