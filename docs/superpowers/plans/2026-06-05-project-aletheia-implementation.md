# Project Aletheia Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the complete Project Aletheia autonomous conversational AI system with Three-Layer Global Memory architecture managed by LangGraph multi-agent society.

**Architecture:** Build the system in layers: 1) Core data models and configuration, 2) Memory storage systems (Neo4j, FAISS, BM25), 3) Offline ingestion pipeline (PDF processing, extraction, conflict resolution), 4) Online orchestration (LangGraph nodes for query routing, retrieval, evaluation, compression, generation), 5) Gradio UI for interaction.

**Tech Stack:** Python, LangGraph, Pydantic, Neo4j, FAISS, rank_bm25, PyMuPDF, transformers, Gradio, Ollama (local LLM), PyTest

---