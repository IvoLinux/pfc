FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# Create necessary directories
RUN mkdir -p models_output/inference \
    && mkdir -p models_output/training \
    && mkdir -p datasets \
    && mkdir -p checkpoints

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]