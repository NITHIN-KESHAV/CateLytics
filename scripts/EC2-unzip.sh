#!/bin/bash

# This script creates an EC2 instance using AWS CLI

# Setting variables for instance creation
AMI_ID="ami-12345678"  
INSTANCE_TYPE="c5.4xlarge"  
KEY_NAME="unzip.pem"  
SECURITY_GROUP="sg-08a27bf3bde82465a"  
SUBNET_ID="subnet-07774e143495ee1ac"  
TAG_NAME="unzip" 

# Creating the EC2 instance
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

# Waiting for the instance to reach the running state
echo "Waiting for instance to be in running state..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

echo "EC2 instance created with ID: $INSTANCE_ID"

