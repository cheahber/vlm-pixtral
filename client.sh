#!/bin/bash

BASE_URL="http://localhost:8000"

# Function to start the processor
start_processor() {
    echo "Starting processor..."
    response=$(curl -s -X POST "$BASE_URL/start" -H "Content-Type: application/json")
    echo "Response: $response"
}

# Function to stop the processor
stop_processor() {
    echo "Stopping processor..."
    response=$(curl -s -X POST "$BASE_URL/stop" -H "Content-Type: application/json")
    echo "Response: $response"
}

# Function to update the video path
update_video_path() {
    local video_path=$1
    echo "Updating video path to: $video_path"
    response=$(curl -s -X POST "$BASE_URL/update-video-path" \
        -H "Content-Type: application/json" \
        -d "{\"video_path\": \"$video_path\"}")
    echo "Response: $response"
}

# Function to update the prompt
update_prompt() {
    local prompt=$1
    echo "Updating prompt to: $prompt"
    response=$(curl -s -X POST "$BASE_URL/update-prompt" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"$prompt\"}")
    echo "Response: $response"
}

# Display menu
while true; do
    echo ""
    echo "Choose an action:"
    echo "1) Start Processor"
    echo "2) Stop Processor"
    echo "3) Update Video Path"
    echo "4) Update Prompt"
    echo "5) Exit"
    read -p "Enter your choice: " choice

    case $choice in
        1)
            start_processor
            ;;
        2)
            stop_processor
            ;;
        3)
            read -p "Enter the video path: " video_path
            update_video_path "$video_path"
            ;;
        4)
            read -p "Enter the prompt: " prompt
            update_prompt "$prompt"
            ;;
        5)
            echo "Exiting script. Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid choice. Please try again."
            ;;
    esac
done
