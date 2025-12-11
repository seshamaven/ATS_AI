-- ============================================
-- CANDIDATE_PROFILE_SCORES TABLE COLUMNS
-- Based on PROFILE_TYPE_RULES in profile_type_utils.py
-- ============================================
-- This SQL script shows all columns that should exist in candidate_profile_scores table
-- Each profile type gets a corresponding score column

-- Core columns (required)
candidate_id INT PRIMARY KEY,
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

-- Profile Type Score Columns (100+ columns)
-- Main Profile Types (First 25)
java_score DECIMAL(10,2) DEFAULT 0.0,
dotnet_score DECIMAL(10,2) DEFAULT 0.0,
python_score DECIMAL(10,2) DEFAULT 0.0,
javascript_score DECIMAL(10,2) DEFAULT 0.0,
fullstack_score DECIMAL(10,2) DEFAULT 0.0,
devops_score DECIMAL(10,2) DEFAULT 0.0,
cloud_infra_score DECIMAL(10,2) DEFAULT 0.0,
data_engineering_score DECIMAL(10,2) DEFAULT 0.0,
data_science_score DECIMAL(10,2) DEFAULT 0.0,
business_intelligence_score DECIMAL(10,2) DEFAULT 0.0,
testing_qa_score DECIMAL(10,2) DEFAULT 0.0,
sap_score DECIMAL(10,2) DEFAULT 0.0,
erp_score DECIMAL(10,2) DEFAULT 0.0,
microsoft_power_platform_score DECIMAL(10,2) DEFAULT 0.0,
rpa_score DECIMAL(10,2) DEFAULT 0.0,
cyber_security_score DECIMAL(10,2) DEFAULT 0.0,
mobile_development_score DECIMAL(10,2) DEFAULT 0.0,
salesforce_score DECIMAL(10,2) DEFAULT 0.0,
low_code_no_code_score DECIMAL(10,2) DEFAULT 0.0,
database_score DECIMAL(10,2) DEFAULT 0.0,
integration_apis_score DECIMAL(10,2) DEFAULT 0.0,
ui_ux_score DECIMAL(10,2) DEFAULT 0.0,
support_score DECIMAL(10,2) DEFAULT 0.0,
business_development_score DECIMAL(10,2) DEFAULT 0.0,

-- Additional Profile Types (50+)
go_golang_score DECIMAL(10,2) DEFAULT 0.0,
ruby_score DECIMAL(10,2) DEFAULT 0.0,
php_score DECIMAL(10,2) DEFAULT 0.0,
rust_score DECIMAL(10,2) DEFAULT 0.0,
scala_score DECIMAL(10,2) DEFAULT 0.0,
c_cpp_score DECIMAL(10,2) DEFAULT 0.0,
react_score DECIMAL(10,2) DEFAULT 0.0,
angular_score DECIMAL(10,2) DEFAULT 0.0,
vue_js_score DECIMAL(10,2) DEFAULT 0.0,
node_js_score DECIMAL(10,2) DEFAULT 0.0,
microservices_score DECIMAL(10,2) DEFAULT 0.0,
serverless_score DECIMAL(10,2) DEFAULT 0.0,
aws_score DECIMAL(10,2) DEFAULT 0.0,
azure_score DECIMAL(10,2) DEFAULT 0.0,
gcp_score DECIMAL(10,2) DEFAULT 0.0,
kubernetes_score DECIMAL(10,2) DEFAULT 0.0,
docker_score DECIMAL(10,2) DEFAULT 0.0,
terraform_score DECIMAL(10,2) DEFAULT 0.0,
ansible_score DECIMAL(10,2) DEFAULT 0.0,
jenkins_score DECIMAL(10,2) DEFAULT 0.0,
gitlab_cicd_score DECIMAL(10,2) DEFAULT 0.0,
github_actions_score DECIMAL(10,2) DEFAULT 0.0,
mongodb_score DECIMAL(10,2) DEFAULT 0.0,
postgresql_score DECIMAL(10,2) DEFAULT 0.0,
mysql_score DECIMAL(10,2) DEFAULT 0.0,
redis_score DECIMAL(10,2) DEFAULT 0.0,
elasticsearch_score DECIMAL(10,2) DEFAULT 0.0,
apache_kafka_score DECIMAL(10,2) DEFAULT 0.0,
apache_spark_score DECIMAL(10,2) DEFAULT 0.0,
hadoop_score DECIMAL(10,2) DEFAULT 0.0,
machine_learning_score DECIMAL(10,2) DEFAULT 0.0,
deep_learning_score DECIMAL(10,2) DEFAULT 0.0,
computer_vision_score DECIMAL(10,2) DEFAULT 0.0,
nlp_score DECIMAL(10,2) DEFAULT 0.0,
blockchain_score DECIMAL(10,2) DEFAULT 0.0,
web3_score DECIMAL(10,2) DEFAULT 0.0,
iot_score DECIMAL(10,2) DEFAULT 0.0,
embedded_systems_score DECIMAL(10,2) DEFAULT 0.0,
game_development_score DECIMAL(10,2) DEFAULT 0.0,
ar_vr_score DECIMAL(10,2) DEFAULT 0.0,
fintech_score DECIMAL(10,2) DEFAULT 0.0,
healthcare_it_score DECIMAL(10,2) DEFAULT 0.0,
ecommerce_score DECIMAL(10,2) DEFAULT 0.0,
content_management_score DECIMAL(10,2) DEFAULT 0.0,
video_streaming_score DECIMAL(10,2) DEFAULT 0.0,
network_engineering_score DECIMAL(10,2) DEFAULT 0.0,
system_administration_score DECIMAL(10,2) DEFAULT 0.0,
graphql_score DECIMAL(10,2) DEFAULT 0.0,
typescript_score DECIMAL(10,2) DEFAULT 0.0,
linux_score DECIMAL(10,2) DEFAULT 0.0,

-- Programming Languages (Additional 50)
swift_score DECIMAL(10,2) DEFAULT 0.0,
kotlin_score DECIMAL(10,2) DEFAULT 0.0,
perl_score DECIMAL(10,2) DEFAULT 0.0,
shell_scripting_score DECIMAL(10,2) DEFAULT 0.0,
powershell_score DECIMAL(10,2) DEFAULT 0.0,
groovy_score DECIMAL(10,2) DEFAULT 0.0,
clojure_score DECIMAL(10,2) DEFAULT 0.0,
erlang_score DECIMAL(10,2) DEFAULT 0.0,
elixir_score DECIMAL(10,2) DEFAULT 0.0,
haskell_score DECIMAL(10,2) DEFAULT 0.0,
fsharp_score DECIMAL(10,2) DEFAULT 0.0,
vb_net_score DECIMAL(10,2) DEFAULT 0.0,
cobol_score DECIMAL(10,2) DEFAULT 0.0,
fortran_score DECIMAL(10,2) DEFAULT 0.0,
assembly_score DECIMAL(10,2) DEFAULT 0.0,
matlab_score DECIMAL(10,2) DEFAULT 0.0,
r_score DECIMAL(10,2) DEFAULT 0.0,
julia_score DECIMAL(10,2) DEFAULT 0.0,
lua_score DECIMAL(10,2) DEFAULT 0.0,
dart_score DECIMAL(10,2) DEFAULT 0.0,
objective_c_score DECIMAL(10,2) DEFAULT 0.0,
delphi_score DECIMAL(10,2) DEFAULT 0.0,
pascal_score DECIMAL(10,2) DEFAULT 0.0,
ada_score DECIMAL(10,2) DEFAULT 0.0,
prolog_score DECIMAL(10,2) DEFAULT 0.0,
lisp_score DECIMAL(10,2) DEFAULT 0.0,
smalltalk_score DECIMAL(10,2) DEFAULT 0.0,
ocaml_score DECIMAL(10,2) DEFAULT 0.0,
racket_score DECIMAL(10,2) DEFAULT 0.0,
crystal_score DECIMAL(10,2) DEFAULT 0.0,
nim_score DECIMAL(10,2) DEFAULT 0.0,
zig_score DECIMAL(10,2) DEFAULT 0.0,
v_score DECIMAL(10,2) DEFAULT 0.0,
d_score DECIMAL(10,2) DEFAULT 0.0,
nix_score DECIMAL(10,2) DEFAULT 0.0,

-- Infrastructure & Cloud Tools (Additional)
terraform_cloud_score DECIMAL(10,2) DEFAULT 0.0,
pulumi_score DECIMAL(10,2) DEFAULT 0.0,
cloudformation_score DECIMAL(10,2) DEFAULT 0.0,
arm_templates_score DECIMAL(10,2) DEFAULT 0.0,
bicep_score DECIMAL(10,2) DEFAULT 0.0,
cdk_score DECIMAL(10,2) DEFAULT 0.0,
serverless_framework_score DECIMAL(10,2) DEFAULT 0.0,
sam_score DECIMAL(10,2) DEFAULT 0.0,
zappa_score DECIMAL(10,2) DEFAULT 0.0,
chalice_score DECIMAL(10,2) DEFAULT 0.0,
vercel_score DECIMAL(10,2) DEFAULT 0.0,
netlify_score DECIMAL(10,2) DEFAULT 0.0,
firebase_score DECIMAL(10,2) DEFAULT 0.0,
supabase_score DECIMAL(10,2) DEFAULT 0.0,
appwrite_score DECIMAL(10,2) DEFAULT 0.0,
hasura_score DECIMAL(10,2) DEFAULT 0.0

-- ============================================
-- TOTAL: ~100+ score columns
-- ============================================
-- Note: Column naming convention:
-- 1. Convert to lowercase
-- 2. Replace spaces with underscores
-- 3. Remove special characters (/, -, (), etc.)
-- 4. Append "_score" suffix
-- 
-- Examples:
-- "Java" → "java_score"
-- ".Net" → "dotnet_score"
-- "Full Stack" → "fullstack_score"
-- "Cloud / Infra" → "cloud_infra_score"
-- "Business Intelligence (BI)" → "business_intelligence_score"
-- "C/C++" → "c_cpp_score"
-- "UI/UX" → "ui_ux_score"
-- "AR/VR" → "ar_vr_score"
-- "Go / Golang" → "go_golang_score"
-- ============================================


