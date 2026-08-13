import "dotenv/config";

import { OpenAI } from "openai";
import { z } from "zod";
import { zodTextFormat } from "openai/helpers/zod.js";

const client = new OpenAI();

async function init() {
  const Steps = z.object({
    explaination: z.string(),
    output: z.string(),
  });

  const OutputFormat = z.object({
    steps: z.array(Steps),
    final_answer: z.string(),
  });

  const result = await client.responses.parse({
    model: "gpt-5.6-luna",
    input: [
      { role: "system", content: "You are good math tutor" },
      {
        role: "user",
        content: "how can I solve 8x + 7 = -23",
      },
    ],
    text: {
      format: zodTextFormat(OutputFormat, "result_format"),
    },
  });

  console.log(result.output_parsed);
}

init();
