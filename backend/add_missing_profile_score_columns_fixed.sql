-- ============================================
-- ADD MISSING COLUMNS TO candidate_profile_scores TABLE
-- Based on PROFILE_TYPE_RULES in profile_type_utils.py
-- ============================================
-- NOTE: MySQL does NOT support "IF NOT EXISTS" for ALTER TABLE ADD COLUMN
-- If a column already exists, you'll get a "Duplicate column name" error
-- You can safely ignore those errors or comment out those lines

USE ats_db;

-- ============================================
-- Main Profile Types (25 columns)
-- Run each ALTER TABLE separately, or combine multiple columns in one statement
-- ============================================

-- Option 1: Add columns one by one (safer, can skip if column exists)
ALTER TABLE candidate_profile_scores ADD COLUMN java_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN dotnet_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN python_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN javascript_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN fullstack_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN devops_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN cloud_infra_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN data_engineering_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN data_science_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN business_intelligence_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN testing_qa_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN sap_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN erp_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN microsoft_power_platform_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN rpa_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN cyber_security_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN mobile_development_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN salesforce_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN low_code_no_code_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN database_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN integration_apis_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN ui_ux_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN support_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN business_development_score DECIMAL(10,2) DEFAULT 0.0;

-- ============================================
-- Additional Profile Types (50+ columns)
-- ============================================
ALTER TABLE candidate_profile_scores ADD COLUMN go_golang_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN ruby_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN php_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN rust_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN scala_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN c_cpp_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN react_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN angular_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN vue_js_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN node_js_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN microservices_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN serverless_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN aws_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN azure_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN gcp_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN kubernetes_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN docker_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN terraform_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN ansible_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN jenkins_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN gitlab_cicd_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN github_actions_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN mongodb_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN postgresql_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN mysql_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN redis_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN elasticsearch_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN apache_kafka_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN apache_spark_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN hadoop_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN machine_learning_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN deep_learning_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN computer_vision_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN nlp_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN blockchain_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN web3_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN iot_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN embedded_systems_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN game_development_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN ar_vr_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN fintech_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN healthcare_it_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN ecommerce_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN content_management_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN video_streaming_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN network_engineering_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN system_administration_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN graphql_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN typescript_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN linux_score DECIMAL(10,2) DEFAULT 0.0;

-- ============================================
-- Programming Languages (Additional 50 columns)
-- ============================================
ALTER TABLE candidate_profile_scores ADD COLUMN swift_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN kotlin_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN perl_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN shell_scripting_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN powershell_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN groovy_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN clojure_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN erlang_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN elixir_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN haskell_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN fsharp_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN vb_net_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN cobol_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN fortran_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN assembly_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN matlab_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN r_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN julia_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN lua_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN dart_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN objective_c_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN delphi_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN pascal_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN ada_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN prolog_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN lisp_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN smalltalk_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN ocaml_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN racket_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN crystal_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN nim_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN zig_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN v_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN d_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN nix_score DECIMAL(10,2) DEFAULT 0.0;

-- ============================================
-- Infrastructure & Cloud Tools
-- ============================================
ALTER TABLE candidate_profile_scores ADD COLUMN terraform_cloud_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN pulumi_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN cloudformation_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN arm_templates_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN bicep_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN cdk_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN serverless_framework_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN sam_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN zappa_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN chalice_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN vercel_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN netlify_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN firebase_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN supabase_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN appwrite_score DECIMAL(10,2) DEFAULT 0.0;
ALTER TABLE candidate_profile_scores ADD COLUMN hasura_score DECIMAL(10,2) DEFAULT 0.0;

-- ============================================
-- Verify columns were added
-- ============================================
SELECT 
    COLUMN_NAME,
    DATA_TYPE,
    COLUMN_DEFAULT,
    IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = 'ats_db'
  AND TABLE_NAME = 'candidate_profile_scores'
  AND COLUMN_NAME LIKE '%_score'
ORDER BY COLUMN_NAME;

-- ============================================
-- IMPORTANT NOTES:
-- ============================================
-- 1. MySQL does NOT support "IF NOT EXISTS" for ALTER TABLE ADD COLUMN
-- 2. If a column already exists, you'll get error: "Duplicate column name 'column_name'"
-- 3. You can safely ignore those errors - it means the column already exists
-- 4. To avoid errors, check existing columns first using the SELECT query above
-- 5. Comment out or remove ALTER TABLE statements for columns that already exist
-- ============================================

