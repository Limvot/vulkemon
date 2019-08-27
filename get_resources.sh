#!/usr/bin/env bash

FOLDER="resources"

mkdir -p "$FOLDER"

ensure_downloaded() {
    url="$1"
    file="$2"
    if [ ! -f "$FOLDER/$file" ]
    then
        echo "curl \"$url\" --output \"$FOLDER/$file\""
        curl "$url" --output "$FOLDER/$file"
    fi
}

ensure_downloaded "https://www.spriters-resource.com/resources/sheets/18/19776.png" "littleroot.png"
ensure_downloaded "https://www.spriters-resource.com/resources/sheets/8/8368.gif" "font.gif"
