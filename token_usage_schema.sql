-- Token Usage Tracking Table for Regulatory RAG System
-- Tracks API token usage across all operations for cost monitoring and optimization

CREATE TABLE `token_usage` (
  `id` int NOT NULL AUTO_INCREMENT,
  `operation_type` varchar(50) NOT NULL COMMENT 'Type of operation: embedding, query_embedding, rag_input, rag_output, etc.',
  `model_name` varchar(100) NOT NULL COMMENT 'OpenAI model used (e.g., text-embedding-ada-002, gpt-4)',
  `input_tokens` int DEFAULT 0 COMMENT 'Number of input tokens used',
  `output_tokens` int DEFAULT 0 COMMENT 'Number of output tokens generated',
  `total_tokens` int DEFAULT 0 COMMENT 'Total tokens used (input + output)',
  `cost_usd` decimal(10,6) DEFAULT 0.000000 COMMENT 'Estimated cost in USD',
  `user_query` text COMMENT 'Original user query (for query operations)',
  `query_relevance` varchar(50) DEFAULT NULL COMMENT 'Query relevance classification',
  `regulatory_domains` json DEFAULT NULL COMMENT 'Identified regulatory domains',
  `context_count` int DEFAULT NULL COMMENT 'Number of context items retrieved',
  `response_length` int DEFAULT NULL COMMENT 'Length of generated response',
  `processing_time_ms` int DEFAULT NULL COMMENT 'Processing time in milliseconds',
  `quality_score` decimal(3,2) DEFAULT NULL COMMENT 'Response quality score',
  `safety_score` decimal(3,2) DEFAULT NULL COMMENT 'Response safety score',
  `session_id` varchar(100) DEFAULT NULL COMMENT 'Session identifier',
  `user_id` varchar(100) DEFAULT NULL COMMENT 'User identifier',
  `request_id` varchar(100) DEFAULT NULL COMMENT 'Unique request identifier',
  `metadata` json DEFAULT NULL COMMENT 'Additional metadata',
  `timestamp` datetime DEFAULT CURRENT_TIMESTAMP COMMENT 'When the operation occurred',
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_operation_type` (`operation_type`),
  KEY `idx_model_name` (`model_name`),
  KEY `idx_timestamp` (`timestamp`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_session_id` (`session_id`),
  KEY `idx_query_relevance` (`query_relevance`),
  KEY `idx_cost` (`cost_usd`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='Token usage tracking for OpenAI API calls';

-- Create indexes for common queries
CREATE INDEX `idx_daily_usage` ON `token_usage` (`timestamp`, `operation_type`);
CREATE INDEX `idx_cost_analysis` ON `token_usage` (`timestamp`, `cost_usd`, `model_name`);
CREATE INDEX `idx_user_usage` ON `token_usage` (`user_id`, `timestamp`);
CREATE INDEX `idx_operation_cost` ON `token_usage` (`operation_type`, `cost_usd`);
