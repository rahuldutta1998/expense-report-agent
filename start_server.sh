#!/bin/zsh
cd ~/expense-report-agent
exec /usr/bin/caffeinate -s ~/expense-report-agent/.venv/bin/python server.py
