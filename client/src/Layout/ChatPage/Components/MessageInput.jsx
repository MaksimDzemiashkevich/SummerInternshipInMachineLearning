import { PaperClipOutlined } from '@ant-design/icons';
import { Attachments, Sender } from '@ant-design/x';
import { Button, Flex, Select } from 'antd';
import React, { useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import "../../../css/MessageInput.css"

const IconStyle = { fontSize: 16 };
const API_BASE = 'http://localhost:8000';
const CHAT_URL = `${API_BASE}/chat`;
const STANDARDS_URL = `${API_BASE}/standards`;
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
    const [standards, setStandards] = useState([]);
    const [selected, setSelected] = useState([]);
    const senderRef = useRef(null);
    const fileInputRef = useRef(null);
    // Контроллер текущего запроса: по нему же сверяем, наш ли ответ пришёл.
    const abortRef = useRef(null);
    // Идентификатор диалога: по нему бэкенд держит файлы этого разговора, поэтому
    // «пересчитай тот файл» работает без повторной загрузки.
    const sessionRef = useRef(
        (typeof crypto !== 'undefined' && crypto.randomUUID)
            ? crypto.randomUUID()
            : `s_${Date.now()}_${Math.random().toString(16).slice(2)}`,
    );
    const [lastFileName, setLastFileName] = useState('');

    // Список опций держим стабильным: новый массив на каждый рендер заставляет Select
    // пересчитывать себя всякий раз, когда в чат приходит сообщение.
    const standardOptions = useMemo(
        () => standards.map((s) => ({ value: s.name, label: s.label })),
        [standards],
    );

    useEffect(() => {
        axios
            .get(STANDARDS_URL)
            .then(({ data }) => setStandards(data?.standards ?? []))
            .catch((error) => console.error('Не удалось получить список стандартов:', error));
    }, []);

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

    /** Кнопка в режиме ожидания — это «стоп»: бросаем запрос и освобождаем ввод. */
    const handleCancel = () => {
        if (!abortRef.current) return;
        abortRef.current.abort();
        abortRef.current = null;
        setIsSending(false);
        onSend({
            key: `msg_${Date.now()}_cancel`,
            role: 'ai',
            content: 'Запрос отменён.',
        });
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
        formData.append('standards', JSON.stringify(selected));
        formData.append('session_id', sessionRef.current);

        const attached = items[0]?.originFileObj;
        if (attached) {
            formData.append('file', attached);
            setLastFileName(attached.name);
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

        const controller = new AbortController();
        abortRef.current = controller;
        // Ответ брошенного запроса может прийти уже после начала следующего — такой
        // результат не наш, и ни ленту, ни состояние кнопки трогать не должен.
        const isCurrent = () => abortRef.current === controller;

        axios
            .post(CHAT_URL, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                signal: controller.signal,
            })
            .then(({ data }) => {
                if (!isCurrent()) return;
                if (data?.session_id) sessionRef.current = data.session_id;
                const reply = {
                    key: `msg_${Date.now()}_reply`,
                    role: 'ai',
                    content: data?.reply ?? '',
                };
                // Агент мог создать несколько файлов за один ход — прикладываем все.
                const produced = (data?.files ?? []).filter((f) => f?.content_base64);
                if (produced.length > 0) {
                    reply.attachments = produced.map((f, index) => {
                        const blob = base64ToBlob(f.content_base64, f.media_type);
                        return {
                            uid: `result_${Date.now()}_${index}`,
                            name: f.name || 'result.xlsx',
                            byte: blob.size,
                            type: blob.type,
                            url: URL.createObjectURL(blob),
                        };
                    });
                }
                onSend(reply);
            })
            .catch((error) => {
                if (axios.isCancel(error) || !isCurrent()) return;  // отмену уже показали
                console.error('Ошибка запроса:', error);
                onSend({
                    key: `msg_${Date.now()}_error`,
                    role: 'ai',
                    content: 'Не удалось получить ответ. Попробуйте ещё раз.',
                });
            })
            .finally(() => {
                if (!isCurrent()) return;
                abortRef.current = null;
                setIsSending(false);
            });
    };

    const canSend = Boolean((value ?? '').trim()) || items.length > 0;

    return (
        <div className="composer">
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                style={{ display: 'none' }}
            />

            <div className="composerBar">
                <span className="composerLabel">Стандарты:</span>
                <Select
                    className="standardsSelect"
                    mode="multiple"
                    size="small"
                    allowClear
                    maxTagCount={1}
                    maxTagPlaceholder={(rest) => `ещё ${rest.length}`}
                    placeholder="все загруженные"
                    value={selected}
                    onChange={setSelected}
                    options={standardOptions}
                />
                {/* Место под подсказку есть всегда: иначе её появление в момент отправки
                    сдвигает строку и Select пересчитывает теги. */}
                <span className="composerHint" title={lastFileName || undefined}>
                    {lastFileName ? `в работе: ${lastFileName}` : ''}
                </span>
            </div>

            <Sender
                className='customSender'
                ref={senderRef}
                value={value}
                onChange={setValue}
                suffix={false}
                placeholder="Спросите что-нибудь или приложите файл..."
                onSubmit={handleSubmit}
                onCancel={handleCancel}
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
                            без сопроводительного текста отправлять нужно. В ожидании на её
                            месте кнопка остановки, её action — onCancel. */}
                        {isSending ? <LoadingButton /> : <SendButton disabled={!canSend} />}
                    </Flex>
                )}
            />
        </div>
    );
};

export default MessageInput;
