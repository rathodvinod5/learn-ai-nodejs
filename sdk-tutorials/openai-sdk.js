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
  const result = await client.responses.parse({
    model: "gpt-4.1-mini",
    text: {
      format: zodTextFormat(outputSchema, "risks"),
    },
    input: `Extract the rist fromt the following document
    
    Document:
    Our companu recently launched a new software platform.
    The platform relies on several third party API that may experience downtime.
    In addition we are storing customer data in the cloude and ther are strict 
    regulatory requirements regrading data privacy and protection.
    Some features are still in beta and could potentially introduce bugs
    that affects user experience.

    Please list any risks you find in the document above.
    `,
  });

  console.log("response: ", result.output_parsed);
}

init();
