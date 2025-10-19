#!/bin/bash

# Caminho da pasta onde o script está
cd /home/ec2-user/Chat-Bot/duckdns

# Variaveis do DuckDNS
DOMAIN="chat-bot-ei-truck"
TOKEN="47bb8583-af2e-4aef-b55a-2dbd9b7531b1" 

# Atualiza o IP no DuckDNS
curl "https://www.duckdns.org/update?domains=$DOMAIN&token=$TOKEN&ip="
