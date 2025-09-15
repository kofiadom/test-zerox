import { BedrockRuntimeClient, InvokeModelCommand, ConverseCommand } from '@aws-sdk/client-bedrock-runtime';
import * as dotenv from 'dotenv';

// Load environment variables
dotenv.config();

interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface ChatResponse {
  message: {
    content: string;
  };
  usage?: {
    input_tokens: number;
    output_tokens: number;
  };
  modelUsed: string;
}

/**
 * Simple Bedrock Test - Based on your BedrockLlmService
 */
class BedrockTest {
  private bedrockClient: BedrockRuntimeClient;
  private modelId: string;
  private temperature: number = 0.7;

  constructor() {
    console.log('🚀 Initializing Bedrock Test...');
    
    // Initialize Bedrock client - same pattern as your BedrockLlmService
    const bedrockConfig = {
      region: process.env.BEDROCK_AWS_REGION || 'eu-west-1',
      credentials: {
        accessKeyId: process.env.BEDROCK_AWS_ACCESS_KEY_ID!,
        secretAccessKey: process.env.BEDROCK_AWS_SECRET_ACCESS_KEY!,
      }
    };

    this.bedrockClient = new BedrockRuntimeClient(bedrockConfig);
    this.modelId = process.env.BEDROCK_MODEL || 'eu.anthropic.claude-3-5-sonnet-20241022-v2:0';
    
    console.log(`✅ Bedrock client initialized`);
    console.log(`   Region: ${bedrockConfig.region}`);
    console.log(`   Model: ${this.modelId}`);
  }

  /**
   * Detect if the current model is Nova or Claude - same logic as your codebase
   */
  private isNovaModel(): boolean {
    return this.modelId.includes('amazon.nova');
  }

  /**
   * Main chat method - mirrors your BedrockLlmService.chat()
   */
  async chat(messages: ChatMessage[]): Promise<ChatResponse> {
    console.log(`\n🤖 Starting inference...`);
    
    if (this.isNovaModel()) {
      console.log(`📡 Using Nova model via Converse API: ${this.modelId}`);
      return await this.chatWithNova(messages);
    } else {
      console.log(`📡 Using Claude model via Invoke API: ${this.modelId}`);
      return await this.chatWithBedrock(messages);
    }
  }

  /**
   * Chat using AWS Bedrock Nova models via Converse API
   * Same implementation as your BedrockLlmService.chatWithNova()
   */
  private async chatWithNova(messages: ChatMessage[]): Promise<ChatResponse> {
    const systemMessage = messages.find(m => m.role === 'system')?.content;
    const conversationMessages = messages.filter(m => m.role !== 'system');

    const command = new ConverseCommand({
      modelId: this.modelId,
      messages: conversationMessages.map(msg => ({
        role: msg.role as 'user' | 'assistant',
        content: [{ text: msg.content }]
      })),
      system: systemMessage ? [{ text: systemMessage }] : undefined,
      inferenceConfig: {
        maxTokens: 4000,
        topP: 0.9,
        temperature: this.temperature
      }
    });

    const response = await this.bedrockClient.send(command);
    
    console.log(`✅ Nova inference completed successfully`);
    
    return {
      message: {
        content: response.output?.message?.content?.[0]?.text || ''
      },
      usage: {
        input_tokens: response.usage?.inputTokens || 0,
        output_tokens: response.usage?.outputTokens || 0
      },
      modelUsed: this.modelId
    };
  }

  /**
   * Chat using AWS Bedrock Claude models via Invoke API
   * Same implementation as your BedrockLlmService.chatWithBedrock()
   */
  private async chatWithBedrock(messages: ChatMessage[]): Promise<ChatResponse> {
    const systemMessage = messages.find(m => m.role === 'system')?.content || '';
    const conversationMessages = messages.filter(m => m.role !== 'system');

    const requestBody = {
      anthropic_version: 'bedrock-2023-05-31',
      max_tokens: 4000,
      temperature: this.temperature,
      system: systemMessage,
      messages: conversationMessages.map(msg => ({
        role: msg.role,
        content: msg.content
      }))
    };

    const command = new InvokeModelCommand({
      modelId: this.modelId,
      contentType: 'application/json',
      accept: 'application/json',
      body: JSON.stringify(requestBody)
    });

    const response = await this.bedrockClient.send(command);
    const responseBody = JSON.parse(new TextDecoder().decode(response.body));

    console.log(`✅ Bedrock Claude inference completed successfully`);

    return {
      message: {
        content: responseBody.content[0].text
      },
      usage: {
        input_tokens: responseBody.usage?.input_tokens || 0,
        output_tokens: responseBody.usage?.output_tokens || 0
      },
      modelUsed: this.modelId
    };
  }
}

/**
 * Main test function
 */
async function runTest() {
  console.log('🧪 Simple Bedrock Inference Test');
  console.log('=================================\n');

  // Check required environment variables
  const requiredEnvVars = [
    'BEDROCK_AWS_ACCESS_KEY_ID',
    'BEDROCK_AWS_SECRET_ACCESS_KEY'
  ];

  const missingVars = requiredEnvVars.filter(varName => !process.env[varName]);
  if (missingVars.length > 0) {
    console.error('❌ Missing required environment variables:');
    missingVars.forEach(varName => console.error(`   - ${varName}`));
    console.error('\nPlease set these variables in your .env file or environment.');
    process.exit(1);
  }

  try {
    // Initialize test client
    const testClient = new BedrockTest();
    
    // Simple test message
    const messages: ChatMessage[] = [
      { role: 'user', content: 'What is the capital of France?' }
    ];

    console.log('📝 Test Message: "What is the capital of France?"');
    
    const startTime = Date.now();
    const response = await testClient.chat(messages);
    const duration = Date.now() - startTime;

    console.log(`\n✅ Test completed successfully (${duration}ms)`);
    console.log(`🤖 Model: ${response.modelUsed}`);
    
    if (response.usage) {
      console.log(`📊 Tokens: ${response.usage.input_tokens} in, ${response.usage.output_tokens} out`);
    }
    
    console.log(`💬 Response: ${response.message.content}`);
    console.log('\n🎉 Bedrock inference is working correctly!');

  } catch (error) {
    console.error(`❌ Test failed: ${error.message}`);
    console.error(`   Stack: ${error.stack?.split('\n')[1]?.trim() || 'No stack trace'}`);
    process.exit(1);
  }
}

// Run test if this script is executed directly
if (require.main === module) {
  runTest().catch(error => {
    console.error('💥 Test crashed:', error);
    process.exit(1);
  });
}

// Export for use as a module
export { BedrockTest, runTest };