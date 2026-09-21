import { PaperClipOutlined } from '@ant-design/icons';
import { Attachments, Sender } from '@ant-design/x';
import { Button, Flex } from 'antd';
import React, { useRef, useState } from 'react';
import axios from 'axios';
import "../../../css/MessageInput.css"

const IconStyle = { fontSize: 16 };
const API_URL = 'http://localhost:8000/chat';
// Сколько последних реплик уходит на бэкенд как контекст разговора.
const HISTORY_LIMIT = 12;

/** Файл приходит в JSON-ответе base64-строкой: разворачиваем её в Blob для скачивания. */
const base64ToBlob = (base64, type) => {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
        bytes[i] = binary.charCodeAt(i);
    }
    return new Blob([bytes], { type });
};

const MessageInput = ({ onSend, history = [] }) => {
    const [value, setValue] = useState('');
    const [items, setItems] = useState([]);
    const [isSending, setIsSending] = useState(false);
    const senderRef = useRef(null);
    const fileInputRef = useRef(null);

    const handleAttachClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = (e) => {
        const file = e.target.files?.[0];
        if (!file) return;

        const newItem = {
            uid: `${file.name}_${Date.now()}`,
            name: file.name,
            size: file.size,
            type: file.type,
            status: 'done',
            originFileObj: file,
            url: URL.createObjectURL(file),
        };

        setItems((prev) => [...prev, newItem]);
        e.target.value = '';
    };

    const handleAttachmentsChange = ({ file, fileList }) => {
        if (file.status === 'removed' && file.url?.startsWith('blob:')) {
            URL.revokeObjectURL(file.url);
        }
        setItems(fileList);
    };

    const handleSubmit = (v) => {
        const text = (v ?? '').trim();
        // Отправлять можно текст без файла (разговор), файл без текста (обработка как раньше)
        // или то и другое сразу — но не пустоту.
        if (!text && items.length === 0) return;

        const formData = new FormData();
        formData.append('message', v ?? '');
        formData.append('history', JSON.stringify(
            history.slice(-HISTORY_LIMIT).map((m) => ({ role: m.role, content: m.content ?? '' })),
        ));
        if (items[0]?.originFileObj) {
            formData.append('file', items[0].originFileObj);
        }

        onSend({
            key: `msg_${Date.now()}`,
            role: 'user',
            content: v ?? '',
            attachments: items.map((item) => ({
                uid: item.uid,
                name: item.name,
                byte: item.size,
                type: item.type,
                url: item.url,
            })),
        });

        setValue('');
        setItems([]);
        setIsSending(true);

        axios
            .post(API_URL, formData, { headers: { 'Content-Type': 'multipart/form-data' } })
            .then(({ data }) => {
                const reply = {
                    key: `msg_${Date.now()}_reply`,
                    role: 'ai',
                    content: data?.reply ?? '',
                };
                if (data?.file?.content_base64) {
                    const blob = base64ToBlob(data.file.content_base64, data.file.media_type);
                    reply.attachments = [{
                        uid: `result_${Date.now()}`,
                        name: data.file.name || 'result.xlsx',
                        byte: blob.size,
                        type: blob.type,
                        url: URL.createObjectURL(blob),
                    }];
                }
                onSend(reply);
            })
            .catch((error) => {
                console.error('Ошибка запроса:', error);
                onSend({
                    key: `msg_${Date.now()}_error`,
                    role: 'ai',
                    content: 'Не удалось получить ответ. Попробуйте ещё раз.',
                });
            })
            .finally(() => {
                setIsSending(false);
            });
    };

    const canSend = Boolean((value ?? '').trim()) || items.length > 0;

    return (
        <>
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                style={{ display: 'none' }}
            />

            <Sender
                className='customSender'
                ref={senderRef}
                value={value}
                onChange={setValue}
                suffix={false}
                placeholder="Спросите что-нибудь или приложите файл..."
                onSubmit={handleSubmit}
                loading={isSending}
                header={
                    <Sender.Header
                        title="Вложения"
                        open={items.length > 0}
                        closable={false}
                        styles={{ content: { padding: 0 } }}
                    >
                        <Attachments
                            beforeUpload={() => false}
                            items={items}
                            onChange={handleAttachmentsChange}
                            getDropContainer={() => senderRef.current?.nativeElement}
                        />
                    </Sender.Header>
                }
                footer={(_, { components: { SendButton, LoadingButton } }) => (
                    <Flex justify="space-between" align="center">
                        <Button
                            style={IconStyle}
                            type="text"
                            icon={<PaperClipOutlined />}
                            onClick={handleAttachClick}
                        />
                        {/* Своя кнопка отправки: штатная блокируется на пустом поле, а файл
                            без сопроводительного текста отправлять нужно. */}
                        {isSending ? <LoadingButton /> : <SendButton disabled={!canSend} />}
                    </Flex>
                )}
            />
        </>
    );
};

export default MessageInput;
