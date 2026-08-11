import { PaperClipOutlined } from '@ant-design/icons';
import { FileCard, Sender } from '@ant-design/x';
import { Button, Flex } from 'antd';
import React, { useRef, useState } from 'react';
import "../../../css/MessageInput.css"

const IconStyle = { fontSize: 16 };

const MessageInput = () => {
    const [value, setValue] = useState('');
    const [attachedFiles, setAttachedFiles] = useState([]);
    const senderRef = useRef(null);
    const fileInputRef = useRef(null);

    const handleAttachClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = (e) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setAttachedFiles((prev) => [
            ...prev,
            {
                uid: `${file.name}_${Date.now()}`,
                name: file.name,
                byte: file.size,
                type: file.type,
            },
        ]);

        e.target.value = '';
    };

    const handleRemoveFile = (item) => {
        setAttachedFiles((prev) => prev.filter((f) => f.uid !== item.uid));
    };

    const handleSubmit = (v) => {
        console.log('Отправлено:', v, 'файлы:', attachedFiles);
        setValue('');
        setAttachedFiles([]);
    };

    return (
        <>
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                style={{ display: 'none' }}
            />

            <Sender className='customSender'
                ref={senderRef}
                value={value}
                onChange={setValue}
                suffix={false}
                placeholder="Введите сообщение..."
                onSubmit={handleSubmit}
                header={
                    attachedFiles.length > 0 && (
                        <div style={{ padding: '8px 12px 0' }}>
                            <FileCard.List
                                items={attachedFiles}
                                size="small"
                                removable
                                onRemove={handleRemoveFile}
                                overflow="scrollX"
                            />
                        </div>
                    )
                }
                footer={(actionNode) => (
                    <Flex justify="space-between" align="center">
                        <Button
                            style={IconStyle}
                            type="text"
                            icon={<PaperClipOutlined />}
                            onClick={handleAttachClick}
                        />
                        {actionNode}
                    </Flex>
                )}
            />
        </>
    );
};

export default MessageInput;