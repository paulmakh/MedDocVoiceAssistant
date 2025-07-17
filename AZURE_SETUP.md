# Azure OpenAI Configuration Guide

This guide will help you set up Azure OpenAI for your RAG application.

## Prerequisites

1. An Azure subscription
2. Access to Azure OpenAI Service
3. Created Azure OpenAI resource

## Step 1: Create Azure OpenAI Resource

1. Go to the Azure Portal (https://portal.azure.com)
2. Click "Create a resource"
3. Search for "Azure OpenAI"
4. Select "Azure OpenAI" and click "Create"
5. Fill in the required information:
   - Subscription: Your Azure subscription
   - Resource group: Create new or use existing
   - Region: Choose a region that supports Azure OpenAI
   - Name: Give your resource a unique name
   - Pricing tier: Select appropriate tier

## Step 2: Deploy Models

After creating the resource, you need to deploy models:

### Deploy Chat Model (GPT-3.5 or GPT-4)
1. Go to your Azure OpenAI resource
2. Click on "Model deployments" in the left menu
3. Click "Create new deployment"
4. Select model: `gpt-35-turbo` or `gpt-4`
5. Give it a deployment name (e.g., "chat-gpt35")
6. Configure capacity settings
7. Click "Create"

### Deploy Embedding Model
1. Click "Create new deployment" again
2. Select model: `text-embedding-ada-002`
3. Give it a deployment name (e.g., "text-embedding")
4. Configure capacity settings
5. Click "Create"

## Step 3: Get Configuration Details

1. Go to your Azure OpenAI resource
2. Click on "Keys and Endpoint" in the left menu
3. Copy the following information:
   - **API Key**: One of the two keys shown
   - **Endpoint**: The endpoint URL

4. Go to "Model deployments" and note:
   - **Chat Deployment Name**: The name you gave your chat model deployment
   - **Embedding Deployment Name**: The name you gave your embedding model deployment

## Step 4: Configure Your Application

Update your `.env` file with the following information:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01

# Azure OpenAI Deployment Names
AZURE_OPENAI_CHAT_DEPLOYMENT=chat-gpt35
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding
```

## Example Configuration

Here's an example of what your `.env` file should look like:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=12345abc67890def12345abc67890def
AZURE_OPENAI_ENDPOINT=https://my-openai-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01

# Azure OpenAI Deployment Names
AZURE_OPENAI_CHAT_DEPLOYMENT=chat-gpt35
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding
```

## Testing Your Configuration

After setting up your configuration, you can test it by running:

```bash
python debug_rag.py
```

This will validate your Azure OpenAI setup and process any PDF files in your documents folder.

## Common Issues

### Authentication Errors
- Double-check your API key
- Ensure your endpoint URL is correct
- Verify you're using the correct API version

### Deployment Not Found
- Check that your deployment names match exactly
- Ensure the deployments are in the "Succeeded" state
- Verify you're using the correct resource

### Rate Limiting
- Azure OpenAI has rate limits based on your pricing tier
- Consider upgrading your tier if you encounter frequent rate limits
- Implement retry logic in production applications

## Best Practices

1. **Security**: Never commit your API keys to version control
2. **Environment Variables**: Always use environment variables for sensitive data
3. **Resource Management**: Monitor your usage to avoid unexpected costs
4. **Model Selection**: Choose the appropriate model for your use case:
   - GPT-3.5-turbo: Faster and cheaper, good for most tasks
   - GPT-4: More capable but slower and more expensive

## Support

If you encounter issues:
1. Check the Azure OpenAI documentation
2. Review the Azure Portal for any service issues
3. Verify your subscription has access to Azure OpenAI Service
