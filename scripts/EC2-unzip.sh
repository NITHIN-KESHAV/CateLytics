#!/bin/bash

# This script creates an EC2 instance using AWS CLI

# Set variables for instance creation
AMI_ID="ami-12345678"  # Replace with your desired AMI ID
INSTANCE_TYPE="c5.4xlarge"  # Instance type (adjusted based on the screenshot)
KEY_NAME="unzip.pem"  # Replace with the name of your key pair
SECURITY_GROUP="sg-08a27bf3bde82465a"  # Updated security group ID based on the screenshot
SUBNET_ID="subnet-07774e143495ee1ac"  # Updated subnet ID based on the screenshot
TAG_NAME="unzip"  # Instance tag name updated based on the screenshot

# Create the EC2 instance
echo "Creating EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type $INSTANCE_TYPE \
  --key-name $KEY_NAME \
  --security-group-ids $SECURITY_GROUP \
  --subnet-id $SUBNET_ID \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG_NAME}]" \
  --query 'Instances[0].InstanceId' \
  --output text)

# Wait for the instance to reach the running state
echo "Waiting for instance to be in running state..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

echo "EC2 instance created with ID: $INSTANCE_ID"

