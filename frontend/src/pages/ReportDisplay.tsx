

const PdfDisplay: React.FC = () => {
    return (
        <iframe 
            src="/path/to/report.pdf" 
            style = {{
                width: '100%',
                height: '100vh',
            }}
     />)
}

export default PdfDisplay