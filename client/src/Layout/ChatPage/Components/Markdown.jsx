import React, { useMemo } from "react";
import { marked } from "marked";
import DOMPurify from "dompurify";
import "../../../css/Markdown.css";

// Ответ агента приходит markdown'ом (таблицы проблемных строк, списки). В тексте бывают значения
// из Excel пользователя, поэтому HTML после разбора чистится DOMPurify — без этого ячейка вида
// <img onerror=...> исполнилась бы в браузере.
marked.setOptions({ gfm: true, breaks: true });

const Markdown = ({ text }) => {
    const html = useMemo(() => DOMPurify.sanitize(marked.parse(text || "")), [text]);
    return <div className="markdown-body" dangerouslySetInnerHTML={{ __html: html }} />;
};

export default Markdown;
