#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Installing Node dependencies..."
npm install

echo "Building Next.js frontend..."
npm run build

echo "Installing Python dependencies..."
pip install -r requirements.txt
