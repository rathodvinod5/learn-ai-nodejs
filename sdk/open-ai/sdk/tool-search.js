import "dotenv/config";
import OpenAI from "openai";

const client = new OpenAI();

function getCustomerProfile(customerId) {
  return `Customer Profile for ${customerId}: Name: John Doe, Tier: Platinum, Status: Active.`;
}

function listOpenOrders(customerId) {
  //   return `Open orders for ${customerId}: Order #1002 (Pending Shipment), Order #1045 (Processing).`;
  return `Open orders for ${customerId}: Order #1002(Open Order)`;
  //   return JSON.stringify({
  //     customer_id: customerId,
  //     open_orders: [
  //       {
  //         order_id: "1002",
  //         status: "Open Order",
  //       },
  //     ],
  //   });
}

function cancelOrder(orderId) {
  return `The customer CUST-104 with order id: #${orderId} is being cancelled.`;
}

const crmNamespace = {
  type: "namespace",
  name: "crm",
  description: "CRM operations",
  tools: [
    {
      type: "function",
      name: "get_customer_profile",
      description: "Fetch a customer profile.",
      parameters: {
        type: "object",
        properties: {
          customer_id: {
            type: "string",
          },
        },
        required: ["customer_id"],
        additionalProperties: false,
      },
    },

    {
      type: "function",
      name: "list_open_orders",
      description: "List all open customer orders.",
      defer_loading: true,
      parameters: {
        type: "object",
        properties: {
          customer_id: {
            type: "string",
          },
        },
        required: ["customer_id"],
        additionalProperties: false,
      },
    },

    // Additional deferred tools purely to demonstrate tool search
    {
      type: "function",
      name: "cancel_order",
      description: "Cancel an order.",
      defer_loading: true,
      parameters: {
        type: "object",
        properties: {
          order_id: {
            type: "string",
          },
        },
        required: ["order_id"],
        additionalProperties: false,
      },
    },

    {
      type: "function",
      name: "refund_order",
      description: "Refund an order.",
      defer_loading: true,
      parameters: {
        type: "object",
        properties: {
          order_id: {
            type: "string",
          },
        },
        required: ["order_id"],
        additionalProperties: false,
      },
    },
  ],
};

async function main() {
  const response = await client.responses.create({
    model: "gpt-5.6",
    input: `Please look up customer CUST-104 and tell me their profile and open orders.`,
    // input: `Customer CUST-104 wants to cancel their only open order.

    //     1. Look up the customer.
    //     2. Find their open orders.
    //     3. Cancel the only open order.
    //     4. Tell me what happened.`,
    tools: [
      crmNamespace,
      {
        type: "tool_search",
      },
    ],
  });

  const toolOutputs = [];

  for (const item of response.output) {
    switch (item.type) {
      case "tool_search_call":
        console.log("\n🔎 Model searched deferred tools.");
        break;

      case "function_call": {
        const { customer_id, order_id } = JSON.parse(item.arguments);
        let output = "";
        console.log("item name: ", item);

        switch (item.name) {
          case "get_customer_profile":
            output = getCustomerProfile(customer_id);
            break;

          case "list_open_orders":
            output = listOpenOrders(customer_id);
            break;

          case "cancel_order":
            output = cancelOrder(order_id);
            break;

          default:
            output = JSON.stringify({
              error: `Unknown tool: ${item.name}`,
            });
        }

        toolOutputs.push({
          type: "function_call_output",
          call_id: item.call_id,
          output: output,
        });

        break;
      }
    }
  }

  // No tools called
  if (toolOutputs.length === 0) {
    console.log("\nNo tool execution required.");
    console.log(response.output_text);
    return;
  }

  const finalResponse = await client.responses.create({
    model: "gpt-5.6",
    previous_response_id: response.id,
    input: toolOutputs,
    tools: [
      crmNamespace,
      {
        type: "tool_search",
      },
    ],
  });

  console.log("FINAL OUTPUT: \n", finalResponse.output_text);
}

main().catch(console.error);
