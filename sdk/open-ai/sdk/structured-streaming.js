import "dotenv/config";

import { OpenAI } from "openai";
import { zodTextFormat } from "openai/helpers/zod";
import { z } from "zod";

const EntitiesSchema = z.object({
  attributes: z.array(z.string()),
  colors: z.array(z.string()),
  animals: z.array(z.string()),
});

const openai = new OpenAI();

async function init() {
  const streams = await openai.responses.create({
    model: "gpt-4.1-mini",
    stream: true,
    input: [
      { role: "user", content: "What's the weather like in Paris today?" },
    ],
    text: {
      format: zodTextFormat(EntitiesSchema, "entities"),
    },
  });

  for await (const event of streams) {
    if (event && event.delta) process.stdout.write(event.delta);
  }
}

init();
