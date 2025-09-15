import { zerox } from "zerox";
import * as fs from "fs";
import * as path from "path";
import * as dotenv from "dotenv";

// Load environment variables
dotenv.config();

const FILES_DIR = "files";
const OUTPUT_DIR = "ocr_output";

async function processFiles() {
  // Check if files directory exists
  if (!fs.existsSync(FILES_DIR)) {
    console.error(`❌ Directory "${FILES_DIR}" does not exist. Please create it and add files to process.`);
    process.exit(1);
  }

  // Create output directory if it doesn't exist
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  // Get all files in the directory
  const files = fs.readdirSync(FILES_DIR).filter(file => {
    const ext = path.extname(file).toLowerCase();
    return ['.pdf', '.docx', '.doc', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'].includes(ext);
  });

  if (files.length === 0) {
    console.log(`ℹ️  No supported files found in "${FILES_DIR}" directory.`);
    return;
  }

  console.log(`📁 Found ${files.length} file(s) to process:`);
  files.forEach(file => console.log(`   - ${file}`));
  console.log();

  // Bedrock credentials
  const credentials = {
    accessKeyId: process.env.BEDROCK_AWS_ACCESS_KEY_ID!,
    secretAccessKey: process.env.BEDROCK_AWS_SECRET_ACCESS_KEY!,
    region: process.env.BEDROCK_AWS_REGION || 'eu-west-1'
  };

  // Check required env vars
  if (!credentials.accessKeyId || !credentials.secretAccessKey) {
    console.error('❌ Missing required environment variables:');
    console.error('   - BEDROCK_AWS_ACCESS_KEY_ID');
    console.error('   - BEDROCK_AWS_SECRET_ACCESS_KEY');
    process.exit(1);
  }

  const model = process.env.BEDROCK_MODEL || 'anthropic.claude-3-5-sonnet-20241022-v2:0';

  console.log('🔧 Configuration:');
  console.log(`   Model: ${model}`);
  console.log(`   Region: ${credentials.region}`);
  console.log();

  // Process each file
  for (const file of files) {
    const filePath = path.join(FILES_DIR, file);
    const outputFileName = `${path.parse(file).name}_ocr.json`;
    const outputPath = path.join(OUTPUT_DIR, outputFileName);

    console.log(`🔍 Processing: ${file}`);

    try {
      const startTime = Date.now();

      const result = await zerox({
        filePath,
        credentials,
        modelProvider: 'BEDROCK',
        model,
        cleanup: true,
        concurrency: 5, // Adjust based on your needs
        maxRetries: 3
      });

      const duration = Date.now() - startTime;

      console.log(`✅ Completed: ${file} (${duration}ms)`);
      console.log(`   Pages: ${result.pages.length}`);
      console.log(`   Input tokens: ${result.inputTokens}`);
      console.log(`   Output tokens: ${result.outputTokens}`);
      console.log(`   Saved to: ${outputPath}`);
      console.log();

      // Save result to file
      fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));

    } catch (error) {
      console.error(`❌ Failed to process ${file}: ${error instanceof Error ? error.message : String(error)}`);
      console.error();
    }
  }

  console.log('🎉 OCR processing completed!');
}

// Run if executed directly
if (require.main === module) {
  processFiles().catch(error => {
    console.error('💥 Script crashed:', error);
    process.exit(1);
  });
}

export { processFiles };