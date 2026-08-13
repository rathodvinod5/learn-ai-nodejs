import "dotenv/config";

import OpenAI from "openai";
import { zodTextFormat } from "openai/helpers/zod.js";
import { z } from "zod";

const client = new OpenAI();

const riskSchema = z.object({
  title: z.string().describe("the actual title for the rist"),
  tags: z.array(z.string()).describe("3-4 tags for this risk"),
  score: z.number().min(1).max(5).describe("risk level out of 5"),
});

const outputSchema = z.object({
  risks: z.array(riskSchema).describe("array of risks"),
});

async function init() {
  const streams = await client.responses.create({
    model: "gpt-4.1-mini",
    stream: true,
    input: [
      {
        role: "user",
        content: "Tell me a story and summary of little red riding hood.",
      },
    ],
  });

  for await (const event of streams) {
    // console.log(event);
    if (event && event.delta) process.stdout.write(event.delta);
  }
}

init();
