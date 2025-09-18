import { ClientV2, InferenceParameters, PathInput } from 'mindee';
import * as fs from 'fs';
import * as path from 'path';

const api_key = "";
const model_id = "";

// Init a new client
const mindee_client = new ClientV2({ apiKey: api_key });

async function processFile(inputPath: string): Promise<string> {
    // Set inference parameters
    const params: InferenceParameters = {
        modelId: model_id,
    };

    // Load a file from disk
    const input_source = new PathInput({ inputPath });

    // Send for processing using polling
    const response = await mindee_client.enqueueAndGetInference(input_source, params);

    // Return the full formatted output like test-mindee.ts
    return response.inference.toString();
}

async function main() {
    const filesDir = './files';
    const markdownDir = './markdown_mindee';

    // Ensure output directory exists
    if (!fs.existsSync(markdownDir)) {
        fs.mkdirSync(markdownDir);
    }

    // Read all files from files directory
    const files = fs.readdirSync(filesDir);

    for (const file of files) {
        const inputPath = path.join(filesDir, file);
        const ext = path.extname(file);
        const baseName = path.basename(file, ext);
        const markdownPath = path.join(markdownDir, `${baseName}_ocr.md`);

        try {
            console.log(`Processing ${file}...`);
            const result = await processFile(inputPath);

            // Save full result as markdown
            fs.writeFileSync(markdownPath, result, 'utf8');
            console.log(`Saved to ${markdownPath}`);
        } catch (error) {
            console.error(`Error processing ${file}:`, error);
        }
    }
}

main().catch(console.error);