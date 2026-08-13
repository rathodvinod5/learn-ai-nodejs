import "dotenv/config";

import { OpenAI } from "openai";

const client = new OpenAI();

async function init() {
  const joke = await client.responses.create({
    model: "gpt-5.6-luna",
    input: "Tell me small joke",
    store: true,
  });

  const joke_reason = await client.responses.create({
    model: "gpt-5.6-luna",
    previous_response_id: joke.id,
    input: [
      {
        role: "user",
        content: "Explain why this is funny??",
      },
    ],
    store: true,
  });

  console.log("JOKE: ", joke.output_text, joke_reason.output_text);
}

init();
