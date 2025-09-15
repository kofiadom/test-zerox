import "dotenv/config";
import { S3Client, ListObjectsV2Command, GetObjectCommand } from "@aws-sdk/client-s3";
import { writeFile, mkdir } from "fs/promises";
import { Readable } from "stream";
import * as path from "path";

const s3 = new S3Client({
  region: process.env.AWS_REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
  },
});

async function streamToBuffer(stream: Readable): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    stream.on("data", (chunk) => chunks.push(chunk));
    stream.on("end", () => resolve(Buffer.concat(chunks)));
    stream.on("error", reject);
  });
}

async function downloadFile(bucket: string, key: string, destFolder: string) {
  console.log(`⬇️  Starting download: ${key}`);

  try {
    const command = new GetObjectCommand({ Bucket: bucket, Key: key });
    const response = await s3.send(command);

    if (!response.Body || !(response.Body instanceof Readable)) {
      throw new Error(`Empty body for ${key}`);
    }

    const fileBuffer = await streamToBuffer(response.Body);

    const destPath = path.join(destFolder, key); // preserves folder structure
    await mkdir(path.dirname(destPath), { recursive: true });
    await writeFile(destPath, fileBuffer);

    console.log(`✅ Downloaded: ${key} -> ${destPath}`);
  } catch (err) {
    console.error(`❌ Failed to download ${key}:`, err);
  }
}

async function downloadFolder(bucket: string, prefix: string, destFolder: string) {
  console.log(`📂 Listing files in bucket "${bucket}" under prefix "${prefix}"...`);

  const listCommand = new ListObjectsV2Command({
    Bucket: bucket,
    Prefix: prefix,
  });

  try {
    const listedObjects = await s3.send(listCommand);

    if (!listedObjects.Contents || listedObjects.Contents.length === 0) {
      console.log(`⚠️  No files found under prefix "${prefix}"`);
      return;
    }

    console.log(`📄 Found ${listedObjects.Contents.length} files. Starting download...`);

    for (const obj of listedObjects.Contents) {
      if (obj.Key) {
        await downloadFile(bucket, obj.Key, destFolder);
      }
    }

    console.log("🎉 All downloads completed!");
  } catch (err) {
    console.error("❌ Error listing objects:", err);
  }
}

// Example usage
(async () => {
  const bucketName = process.env.S3_BUCKET!;
  const folderPrefix = "Indonesia/"; // must end with "/" if you want folder-like behavior
  const localSaveDir = "./downloads";

  await downloadFolder(bucketName, folderPrefix, localSaveDir);
})();
