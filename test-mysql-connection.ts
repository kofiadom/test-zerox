/* eslint-disable no-console */
import * as mysql from 'mysql2/promise';
import * as dotenv from 'dotenv';

// Load environment variables
dotenv.config();

async function testMySQLConnection() {
  let connection: mysql.Connection | null = null;

  try {
    console.log('Attempting to connect to MySQL...');

    // Create connection using environment variables
    connection = await mysql.createConnection({
      host: process.env.MYSQL_HOST,
      port: parseInt(process.env.MYSQL_PORT || '3306', 10),
      user: process.env.MYSQL_USER,
      password: process.env.MYSQL_PASSWORD,
      database: process.env.MYSQL_DATABASE,
      ssl: process.env.MYSQL_SSL === 'true' ? { rejectUnauthorized: false } : undefined,
    });

    console.log('✅ Successfully connected to MySQL!');

    // Test a simple query
    const [rows] = await connection.execute('SELECT 1 as test');
    console.log('✅ Test query executed successfully:', rows);
  } catch (error) {
    console.error('❌ Failed to connect to MySQL:', error);
  } finally {
    if (connection) {
      await connection.end();
      console.log('🔌 Connection closed.');
    }
  }
}

// Run the test
testMySQLConnection();
