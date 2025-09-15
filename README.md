# OCR Multiple Files Script

This script uses Zerox to perform OCR (Optical Character Recognition) on multiple files in a directory using AWS Bedrock models.

## Prerequisites

- Node.js (v14 or higher)
- Ghostscript (for PDF processing)
- AWS Bedrock access with appropriate permissions

## Installation

1. **Install Ghostscript** (required for PDF processing):
   - Download from: https://www.ghostscript.com/download.html
   - Install and ensure it's available in your system PATH

2. **Install dependencies**:
   ```bash
   npm install
   ```

## Environment Setup

Create a `.env` file in the project root with your AWS Bedrock credentials:

```env
BEDROCK_AWS_ACCESS_KEY_ID=your_access_key
BEDROCK_AWS_SECRET_ACCESS_KEY=your_secret_key
BEDROCK_AWS_REGION=eu-west-1
BEDROCK_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0
```

## Usage

1. **Prepare your files**:
   - Create a `files` directory in the project root
   - Add files to process (supported: PDF, DOCX, DOC, PNG, JPG, JPEG, TIFF, BMP)

2. **Run the script**:
   ```bash
   npx ts-node ocr-multiple-files.ts
   ```

## Output

- Results are saved in the `ocr_output` directory as JSON files
- Each file gets its own output file (e.g., `document.pdf` → `document_ocr.json`)
- Output includes:
  - Extracted text content per page
  - Processing time and token usage
  - Success/failure status

## Supported Models

The script uses AWS Bedrock models. Default is `anthropic.claude-3-5-sonnet-20241022-v2:0`. You can change this in your `.env` file.

Other supported models include:
- `anthropic.claude-3-5-haiku-20241022-v1:0`
- `anthropic.claude-3-haiku-20240307-v1:0`
- `anthropic.claude-3-opus-20240229-v1:0`

## Configuration

You can modify the script for:
- **Concurrency**: Change the `concurrency` parameter (default: 5)
- **Retries**: Adjust `maxRetries` (default: 3)
- **Output directory**: Modify `OUTPUT_DIR` variable

## Troubleshooting

- **"Ghostscript not found"**: Ensure Ghostscript is installed and in PATH
- **AWS credentials error**: Check your `.env` file and AWS permissions
- **No files found**: Ensure files are in the `files` directory with supported extensions

## Example Output Structure

```json
{
  "completionTime": 10038,
  "fileName": "document",
  "inputTokens": 25543,
  "outputTokens": 210,
  "pages": [
    {
      "page": 1,
      "content": "# Extracted Text Content...",
      "contentLength": 747
    }
  ],
  "summary": {
    "totalPages": 1,
    "ocr": {
      "failed": 0,
      "successful": 1
    }
  }
}